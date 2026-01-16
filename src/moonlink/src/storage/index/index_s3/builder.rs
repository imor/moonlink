//! Builder for creating and merging S3 indices.
//!
//! This module provides `S3GlobalIndexBuilder` for constructing new indices
//! from hash entries or merging existing indices.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};

use super::client::S3Client;
use super::config::S3IndexConfig;
use super::entry::S3IndexEntry;
use super::error::{S3IndexError, S3IndexResult};
use super::format::{
    encode_file_list, BucketInfo, S3IndexHeader, BUCKET_DIR_ENTRY_SIZE, CURRENT_VERSION,
    HEADER_SIZE,
};
use super::index::S3GlobalIndex;
use super::reader::splitmix64;

/// Builder for creating S3 indices.
///
/// `S3GlobalIndexBuilder` supports two modes of index creation:
///
/// 1. **From flush**: Create a new index from hash entries (during data flush)
/// 2. **From merge**: Combine multiple existing indices into one
///
/// # Example: Build from flush
///
/// ```rust,ignore
/// let builder = S3GlobalIndexBuilder::new(s3_client, config)
///     .set_files(vec!["data1.parquet".into(), "data2.parquet".into()]);
///
/// // hash_entries: Vec<(key_hash, file_idx, row_idx)>
/// let index = builder.build_from_flush(hash_entries, index_id).await?;
/// ```
///
/// # Example: Build from merge
///
/// ```rust,ignore
/// let builder = S3GlobalIndexBuilder::new(s3_client, config);
/// let merged = builder
///     .build_from_merge(&[&index1, &index2, &index3], new_index_id)
///     .await?;
/// ```
pub struct S3GlobalIndexBuilder {
    /// S3 client for uploading the index.
    s3_client: Arc<dyn S3Client>,
    /// Index configuration.
    config: S3IndexConfig,
    /// Data files covered by this index.
    files: Vec<String>,
}

impl S3GlobalIndexBuilder {
    /// Create a new builder.
    pub fn new(s3_client: Arc<dyn S3Client>, config: S3IndexConfig) -> Self {
        Self {
            s3_client,
            config,
            files: Vec::new(),
        }
    }

    /// Set the list of data files covered by this index.
    pub fn set_files(mut self, files: Vec<String>) -> Self {
        self.files = files;
        self
    }

    /// Build index from hash entries (during data flush).
    ///
    /// # Arguments
    ///
    /// * `hash_entries` - Vector of (hash_key, file_idx, row_idx) tuples
    ///   - `hash_key`: The pre-computed hash of the primary key
    ///   - `file_idx`: Index into the files list
    ///   - `row_idx`: Row number within the file
    /// * `index_id` - Unique identifier for this index
    ///
    /// # Returns
    ///
    /// The created `S3GlobalIndex` ready for lookups.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let entries = vec![
    ///     (splitmix64(100), 0, 0),   // key=100 in file[0], row 0
    ///     (splitmix64(200), 0, 1),   // key=200 in file[0], row 1
    ///     (splitmix64(300), 1, 0),   // key=300 in file[1], row 0
    /// ];
    ///
    /// let index = builder.build_from_flush(entries, 1).await?;
    /// ```
    pub async fn build_from_flush(
        self,
        hash_entries: Vec<(u64, u32, u64)>,
        index_id: u64,
    ) -> S3IndexResult<S3GlobalIndex> {
        // Calculate bit widths based on data
        let max_file_idx = hash_entries
            .iter()
            .map(|e| e.1)
            .max()
            .unwrap_or(0)
            .max(self.files.len().saturating_sub(1) as u32);
        let max_row_idx = hash_entries.iter().map(|e| e.2).max().unwrap_or(0);

        let seg_id_bits = bits_needed(max_file_idx as u64).max(1);
        let row_id_bits = bits_needed(max_row_idx).max(1);

        // Use 32 bits for bucket selection, 32 for storage
        let hash_upper_bits = 32;
        let hash_lower_bits = 32;

        let entry_size = S3IndexHeader::calculate_entry_size(hash_lower_bits, seg_id_bits, row_id_bits);

        // Distribute entries into buckets
        let num_buckets = self.config.num_buckets;
        let mut buckets: Vec<Vec<S3IndexEntry>> = vec![Vec::new(); num_buckets as usize];

        for (hash, file_idx, row_idx) in hash_entries {
            let lower_mask = (1u64 << hash_lower_bits) - 1;
            let lower_hash = hash & lower_mask;
            let bucket_idx = ((hash >> hash_lower_bits) as u32) % num_buckets;

            buckets[bucket_idx as usize].push(S3IndexEntry {
                lower_hash,
                file_idx,
                row_idx,
            });
        }

        // Sort entries within each bucket by lower_hash for better lookup performance
        for bucket in &mut buckets {
            bucket.sort_by_key(|e| e.lower_hash);
        }

        // Build the index file
        let index_data = self.build_index_bytes(
            &buckets,
            hash_upper_bits,
            hash_lower_bits,
            seg_id_bits,
            row_id_bits,
            entry_size,
        )?;

        // Upload to S3
        let s3_key = self.config.index_key(index_id);
        self.s3_client
            .put_object(&s3_key, index_data)
            .await?;

        // Return the index
        Ok(S3GlobalIndex::open(
            s3_key,
            self.s3_client,
            self.config,
            index_id,
        ))
    }

    /// Build index from raw key-value pairs (convenience method).
    ///
    /// This method hashes the keys internally.
    ///
    /// # Arguments
    ///
    /// * `key_entries` - Vector of (key, file_idx, row_idx) tuples
    /// * `index_id` - Unique identifier for this index
    pub async fn build_from_keys(
        self,
        key_entries: Vec<(u64, u32, u64)>,
        index_id: u64,
    ) -> S3IndexResult<S3GlobalIndex> {
        let hash_entries: Vec<_> = key_entries
            .into_iter()
            .map(|(key, file_idx, row_idx)| (splitmix64(key), file_idx, row_idx))
            .collect();

        self.build_from_flush(hash_entries, index_id).await
    }

    /// Build index by merging existing indices.
    ///
    /// This creates a new index containing all entries from the input indices,
    /// with file indices remapped to the combined file list.
    ///
    /// # Arguments
    ///
    /// * `indices` - Slice of indices to merge
    /// * `index_id` - Unique identifier for the merged index
    ///
    /// # Returns
    ///
    /// The merged `S3GlobalIndex`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // After accumulating small indices
    /// let merged = builder
    ///     .build_from_merge(&[&small_idx1, &small_idx2, &small_idx3], new_id)
    ///     .await?;
    ///
    /// // Old indices can now be deleted
    /// small_idx1.delete().await?;
    /// small_idx2.delete().await?;
    /// small_idx3.delete().await?;
    /// ```
    pub async fn build_from_merge(
        mut self,
        indices: &[&S3GlobalIndex],
        index_id: u64,
    ) -> S3IndexResult<S3GlobalIndex> {
        if indices.is_empty() {
            return Err(S3IndexError::invalid_format("Cannot merge empty index list"));
        }

        // Collect all files and build mapping
        let mut file_mapping: HashMap<(usize, u32), u32> = HashMap::new(); // (index_idx, old_file_idx) -> new_file_idx
        let mut all_files: Vec<String> = Vec::new();

        for (idx_num, index) in indices.iter().enumerate() {
            let files = index.files().await?;
            for (old_idx, file) in files.into_iter().enumerate() {
                let new_idx = all_files.len() as u32;
                file_mapping.insert((idx_num, old_idx as u32), new_idx);
                all_files.push(file);
            }
        }

        self.files = all_files;

        // Collect all entries with remapped file indices
        let mut all_hash_entries: Vec<(u64, u32, u64)> = Vec::new();

        for (idx_num, index) in indices.iter().enumerate() {
            let header = index.header().await?;

            // Read all buckets from this index
            for bucket_idx in 0..header.num_buckets {
                // Reconstruct hash and get entries
                // Note: This is somewhat inefficient as we're reading all entries
                // A production implementation might want to stream entries directly
                let entries = read_bucket_entries_for_merge(index, bucket_idx).await?;

                for entry in entries {
                    // Reconstruct the full hash (approximately)
                    let hash = ((bucket_idx as u64) << header.hash_lower_bits) | entry.lower_hash;

                    // Remap file index
                    let new_file_idx = file_mapping
                        .get(&(idx_num, entry.file_idx))
                        .copied()
                        .ok_or_else(|| {
                            S3IndexError::internal(format!(
                                "File mapping not found for index {} file {}",
                                idx_num, entry.file_idx
                            ))
                        })?;

                    all_hash_entries.push((hash, new_file_idx, entry.row_idx));
                }
            }
        }

        // Build the merged index
        self.build_from_flush(all_hash_entries, index_id).await
    }

    /// Build the index file bytes.
    fn build_index_bytes(
        &self,
        buckets: &[Vec<S3IndexEntry>],
        hash_upper_bits: u32,
        hash_lower_bits: u32,
        seg_id_bits: u32,
        row_id_bits: u32,
        entry_size: u32,
    ) -> S3IndexResult<Bytes> {
        let num_buckets = buckets.len() as u32;
        let num_entries: u64 = buckets.iter().map(|b| b.len() as u64).sum();

        // Encode file list
        let file_list_data = encode_file_list(&self.files)?;

        // Calculate section sizes and offsets
        let file_list_offset = HEADER_SIZE;
        let file_list_size = file_list_data.len() as u64;

        let bucket_dir_offset = file_list_offset + file_list_size;
        let bucket_dir_size = (num_buckets as u64 + 1) * BUCKET_DIR_ENTRY_SIZE;

        let entry_block_offset = bucket_dir_offset + bucket_dir_size;

        // Build bucket directory and entries
        let mut bucket_infos: Vec<BucketInfo> = Vec::with_capacity(num_buckets as usize + 1);
        let mut entry_offset: u64 = 0;

        // Create header for encoding entries
        let header = S3IndexHeader {
            version: CURRENT_VERSION,
            num_buckets,
            num_entries,
            num_files: self.files.len() as u32,
            hash_upper_bits,
            hash_lower_bits,
            seg_id_bits,
            row_id_bits,
            entry_size,
            bucket_dir_offset,
            bucket_dir_size,
            entry_block_offset,
            entry_block_size: 0, // Will be calculated
            file_list_offset,
            file_list_size,
            checksum: 0, // Will be calculated during encode
        };

        // Build entries and track offsets
        let mut all_entries = BytesMut::new();
        for bucket in buckets {
            bucket_infos.push(BucketInfo { entry_offset });
            let bucket_bytes = S3IndexEntry::encode_all(bucket, &header);
            entry_offset += bucket_bytes.len() as u64;
            all_entries.put(bucket_bytes);
        }
        // Sentinel entry for last bucket's end offset
        bucket_infos.push(BucketInfo { entry_offset });

        let entry_block_size = entry_offset;

        // Build final header with correct sizes
        let final_header = S3IndexHeader {
            entry_block_size,
            ..header
        };

        // Assemble the file
        let total_size = entry_block_offset + entry_block_size;
        let mut output = BytesMut::with_capacity(total_size as usize);

        // 1. Header
        output.put(final_header.encode());

        // 2. File list
        output.put(file_list_data);

        // 3. Bucket directory
        output.put(BucketInfo::encode_directory(&bucket_infos));

        // 4. Entries
        output.put(all_entries);

        Ok(output.freeze())
    }
}

/// Calculate number of bits needed to represent a value.
fn bits_needed(value: u64) -> u32 {
    if value == 0 {
        1
    } else {
        64 - value.leading_zeros()
    }
}

/// Read all entries from a bucket for merging.
///
/// This is a helper function that fetches entries without using the cache,
/// since we're reading the entire index anyway during merge.
async fn read_bucket_entries_for_merge(
    index: &S3GlobalIndex,
    bucket_idx: u32,
) -> S3IndexResult<Vec<S3IndexEntry>> {
    use super::reader::fetch_bucket_entries;

    let reader = &index.reader;
    reader.ensure_loaded().await?;

    let guard = reader.metadata.read().await;
    let metadata = guard.as_ref().ok_or_else(|| {
        S3IndexError::internal("Metadata not loaded")
    })?;

    fetch_bucket_entries(
        reader.s3_client.as_ref(),
        &reader.s3_key,
        metadata,
        bucket_idx,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_needed() {
        assert_eq!(bits_needed(0), 1);
        assert_eq!(bits_needed(1), 1);
        assert_eq!(bits_needed(2), 2);
        assert_eq!(bits_needed(3), 2);
        assert_eq!(bits_needed(4), 3);
        assert_eq!(bits_needed(255), 8);
        assert_eq!(bits_needed(256), 9);
        assert_eq!(bits_needed(u32::MAX as u64), 32);
    }
}
