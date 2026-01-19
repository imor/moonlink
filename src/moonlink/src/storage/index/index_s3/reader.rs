//! Reader utilities for S3 index lookup operations.
//!
//! This module provides functions for reading index data from S3 and
//! performing lookups.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use super::cache::{CacheKey, CachedBucket, S3IndexCache};
use super::client::S3Client;
use super::entry::S3IndexEntry;
use super::error::{S3IndexError, S3IndexResult};
use super::format::{decode_file_list, BucketInfo, S3IndexHeader, HEADER_SIZE};

/// High-quality hash function for distributing keys across buckets.
///
/// SplitMix64 is a fast, non-cryptographic hash function with excellent
/// statistical properties.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Loaded index metadata (always kept in memory after first access).
pub(crate) struct LoadedIndexMetadata {
    /// Index header.
    pub header: S3IndexHeader,
    /// Bucket directory (offset for each bucket).
    pub bucket_directory: Vec<BucketInfo>,
    /// List of data files covered by this index.
    pub files: Vec<String>,
}

/// Load index metadata from S3 (header, bucket directory, file list).
///
/// This fetches all metadata needed for lookups in a minimal number of requests.
pub(crate) async fn load_index_metadata(
    s3_client: &dyn S3Client,
    s3_key: &str,
) -> S3IndexResult<LoadedIndexMetadata> {
    // Step 1: Fetch header
    let header_bytes = s3_client.get_range(s3_key, 0..HEADER_SIZE).await?;
    let header = S3IndexHeader::decode(&header_bytes)?;

    // Step 2: Fetch file list (usually small, right after header)
    let file_list_bytes = s3_client
        .get_range(
            s3_key,
            header.file_list_offset..(header.file_list_offset + header.file_list_size),
        )
        .await?;
    let files = decode_file_list(&file_list_bytes)?;

    // Step 3: Fetch bucket directory
    let bucket_dir_bytes = s3_client
        .get_range(
            s3_key,
            header.bucket_dir_offset..(header.bucket_dir_offset + header.bucket_dir_size),
        )
        .await?;
    let bucket_directory = BucketInfo::decode_directory(&bucket_dir_bytes, header.num_buckets)?;

    Ok(LoadedIndexMetadata {
        header,
        bucket_directory,
        files,
    })
}

/// Fetch entries for a specific bucket from S3.
pub(crate) async fn fetch_bucket_entries(
    s3_client: &dyn S3Client,
    s3_key: &str,
    metadata: &LoadedIndexMetadata,
    bucket_idx: u32,
) -> S3IndexResult<Vec<S3IndexEntry>> {
    if bucket_idx >= metadata.header.num_buckets {
        return Err(S3IndexError::invalid_format(format!(
            "Bucket index {} out of range (max {})",
            bucket_idx, metadata.header.num_buckets
        )));
    }

    // Get bucket boundaries from directory
    let bucket_array_offset = metadata.bucket_directory[bucket_idx as usize].entry_offset;
    let bucket_end_offset = metadata.bucket_directory[bucket_idx as usize + 1].entry_offset;

    // Empty bucket
    if bucket_array_offset == bucket_end_offset {
        return Ok(Vec::new());
    }

    // Calculate absolute byte range
    let abs_start = metadata.header.entry_block_offset + bucket_array_offset;
    let abs_end = metadata.header.entry_block_offset + bucket_end_offset;

    // Fetch entry data
    let entry_bytes = s3_client.get_range(s3_key, abs_start..abs_end).await?;

    // Decode entries
    S3IndexEntry::decode_all(&entry_bytes, &metadata.header)
}

/// Fetch entries for multiple buckets, combining adjacent ranges.
///
/// This optimizes S3 requests by combining adjacent or nearby bucket fetches
/// into single range requests.
pub(crate) async fn fetch_bucket_entries_batch(
    s3_client: &dyn S3Client,
    s3_key: &str,
    metadata: &LoadedIndexMetadata,
    bucket_idxs: &[u32],
) -> S3IndexResult<HashMap<u32, Vec<S3IndexEntry>>> {
    if bucket_idxs.is_empty() {
        return Ok(HashMap::new());
    }

    let mut results = HashMap::new();

    // For now, fetch each bucket separately
    // TODO: Optimize by combining adjacent ranges into single requests
    for &bucket_idx in bucket_idxs {
        let entries = fetch_bucket_entries(s3_client, s3_key, metadata, bucket_idx).await?;
        results.insert(bucket_idx, entries);
    }

    Ok(results)
}

/// Filter entries by lower hash and return matching record locations.
pub(crate) fn filter_entries_by_hash(
    entries: &[S3IndexEntry],
    lower_hash: u64,
    files: &[String],
) -> Vec<RecordLocation> {
    entries
        .iter()
        .filter(|e| e.lower_hash == lower_hash)
        .map(|e| RecordLocation {
            file_path: files[e.file_idx as usize].clone(),
            row_idx: e.row_idx,
        })
        .collect()
}

/// Location of a record in a data file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordLocation {
    /// Path to the data file (e.g., S3 URI or Iceberg file path).
    pub file_path: String,
    /// Row index within the data file.
    pub row_idx: u64,
}

/// Shared index reader that can be used concurrently.
///
/// Wraps the metadata and cache with proper synchronization.
pub(crate) struct SharedIndexReader {
    /// S3 key for the index file.
    pub s3_key: String,
    /// S3 client.
    pub s3_client: Arc<dyn S3Client>,
    /// Loaded metadata (lazily initialized).
    pub metadata: RwLock<Option<LoadedIndexMetadata>>,
    /// Entry cache.
    pub cache: RwLock<S3IndexCache>,
}

impl SharedIndexReader {
    /// Create a new shared reader.
    pub fn new(s3_key: String, s3_client: Arc<dyn S3Client>, cache: S3IndexCache) -> Self {
        Self {
            s3_key,
            s3_client,
            metadata: RwLock::new(None),
            cache: RwLock::new(cache),
        }
    }

    /// Ensure metadata is loaded.
    pub async fn ensure_loaded(&self) -> S3IndexResult<()> {
        // Fast path: already loaded
        {
            let guard = self.metadata.read().await;
            if guard.is_some() {
                return Ok(());
            }
        }

        // Slow path: need to load
        let mut guard = self.metadata.write().await;
        // Double-check after acquiring write lock
        if guard.is_some() {
            return Ok(());
        }

        let loaded = load_index_metadata(self.s3_client.as_ref(), &self.s3_key).await?;
        *guard = Some(loaded);
        Ok(())
    }

    /// Get the header (must call ensure_loaded first).
    pub async fn header(&self) -> S3IndexResult<S3IndexHeader> {
        let guard = self.metadata.read().await;
        guard
            .as_ref()
            .map(|m| m.header.clone())
            .ok_or_else(|| S3IndexError::internal("Metadata not loaded"))
    }

    /// Get the file list (must call ensure_loaded first).
    pub async fn files(&self) -> S3IndexResult<Vec<String>> {
        let guard = self.metadata.read().await;
        guard
            .as_ref()
            .map(|m| m.files.clone())
            .ok_or_else(|| S3IndexError::internal("Metadata not loaded"))
    }

    /// Find records matching a key.
    pub async fn find(&self, key: u64) -> S3IndexResult<Vec<RecordLocation>> {
        self.ensure_loaded().await?;

        let hash = splitmix64(key);

        // Get header info
        let (num_buckets, hash_lower_bits) = {
            let guard = self.metadata.read().await;
            let metadata = guard.as_ref().unwrap();
            (metadata.header.num_buckets, metadata.header.hash_lower_bits)
        };

        // Calculate bucket and lower hash
        let lower_mask = (1u64 << hash_lower_bits) - 1;
        let lower_hash = hash & lower_mask;
        let bucket_idx = ((hash >> hash_lower_bits) as u32) % num_buckets;

        // Check cache
        let cache_key = CacheKey::new(&self.s3_key, bucket_idx);
        {
            let mut cache = self.cache.write().await;
            if let Some(cached) = cache.get(&cache_key) {
                let files = self.files().await?;
                return Ok(filter_entries_by_hash(&cached.entries, lower_hash, &files));
            }
        }

        // Fetch from S3
        let entries = {
            let guard = self.metadata.read().await;
            let metadata = guard.as_ref().unwrap();
            fetch_bucket_entries(self.s3_client.as_ref(), &self.s3_key, metadata, bucket_idx)
                .await?
        };

        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, CachedBucket::new(entries.clone()));
        }

        // Filter and return
        let files = self.files().await?;
        Ok(filter_entries_by_hash(&entries, lower_hash, &files))
    }

    /// Batch find for multiple keys (more efficient).
    pub async fn find_batch(&self, keys: &[u64]) -> S3IndexResult<Vec<(u64, Vec<RecordLocation>)>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_loaded().await?;

        // Get header info
        let (num_buckets, hash_lower_bits) = {
            let guard = self.metadata.read().await;
            let metadata = guard.as_ref().unwrap();
            (metadata.header.num_buckets, metadata.header.hash_lower_bits)
        };

        let lower_mask = (1u64 << hash_lower_bits) - 1;

        // Group keys by bucket
        let mut bucket_to_keys: HashMap<u32, Vec<(u64, u64)>> = HashMap::new(); // bucket -> [(key, lower_hash)]
        for &key in keys {
            let hash = splitmix64(key);
            let lower_hash = hash & lower_mask;
            let bucket_idx = ((hash >> hash_lower_bits) as u32) % num_buckets;
            bucket_to_keys
                .entry(bucket_idx)
                .or_default()
                .push((key, lower_hash));
        }

        // Find buckets not in cache
        let mut buckets_to_fetch = Vec::new();
        {
            let cache = self.cache.read().await;
            for &bucket_idx in bucket_to_keys.keys() {
                let cache_key = CacheKey::new(&self.s3_key, bucket_idx);
                if !cache.contains(&cache_key) {
                    buckets_to_fetch.push(bucket_idx);
                }
            }
        }

        // Fetch missing buckets
        if !buckets_to_fetch.is_empty() {
            let fetched = {
                let guard = self.metadata.read().await;
                let metadata = guard.as_ref().unwrap();
                fetch_bucket_entries_batch(
                    self.s3_client.as_ref(),
                    &self.s3_key,
                    metadata,
                    &buckets_to_fetch,
                )
                .await?
            };

            // Cache fetched buckets
            let mut cache = self.cache.write().await;
            for (bucket_idx, entries) in fetched {
                let cache_key = CacheKey::new(&self.s3_key, bucket_idx);
                cache.insert(cache_key, CachedBucket::new(entries));
            }
        }

        // Now all needed buckets are in cache - collect results
        let files = self.files().await?;
        let mut results = Vec::with_capacity(keys.len());

        let mut cache = self.cache.write().await;
        for (bucket_idx, key_hashes) in bucket_to_keys {
            let cache_key = CacheKey::new(&self.s3_key, bucket_idx);
            if let Some(cached) = cache.get(&cache_key) {
                for (key, lower_hash) in key_hashes {
                    let locations = filter_entries_by_hash(&cached.entries, lower_hash, &files);
                    results.push((key, locations));
                }
            }
        }

        Ok(results)
    }
}
