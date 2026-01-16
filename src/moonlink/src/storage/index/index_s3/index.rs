//! S3 Global Index - the main index structure.
//!
//! This module provides `S3GlobalIndex`, the primary interface for S3-based
//! index operations.

use std::sync::Arc;

use super::cache::{S3IndexCache, S3IndexCacheConfig};
use super::client::S3Client;
use super::config::S3IndexConfig;
use super::error::S3IndexResult;
use super::format::S3IndexHeader;
use super::reader::{RecordLocation, SharedIndexReader};

/// S3-based global index for fast primary key lookups.
///
/// `S3GlobalIndex` provides efficient lookups of records by primary key,
/// with data stored entirely on S3. It uses:
/// - **Lazy loading**: Metadata is fetched on first access
/// - **LRU caching**: Bucket entries are cached in memory
/// - **Range requests**: Only needed data is fetched from S3
///
/// # Thread Safety
///
/// `S3GlobalIndex` is safe to share across threads. Internal state is protected
/// by `RwLock` for concurrent read access with exclusive write access.
///
/// # Example
///
/// ```rust,ignore
/// // Create index from S3
/// let index = S3GlobalIndex::open(
///     "indices/table_123/index_001.bin",
///     s3_client,
///     S3IndexConfig::new("my-bucket", "indices/table_123")
///         .with_cache_size(100 * 1024 * 1024),
/// );
///
/// // Single key lookup
/// let locations = index.find(12345).await?;
/// for loc in locations {
///     println!("Found at file={}, row={}", loc.file_path, loc.row_idx);
/// }
///
/// // Batch lookup (more efficient)
/// let results = index.find_batch(&[100, 200, 300]).await?;
/// ```
#[derive(Clone)]
pub struct S3GlobalIndex {
    /// Shared reader state.
    pub(crate) reader: Arc<SharedIndexReader>,
    /// Configuration.
    config: S3IndexConfig,
    /// Index ID (unique identifier).
    index_id: u64,
}

impl S3GlobalIndex {
    /// Create a new index reference for an existing S3 index file.
    ///
    /// This does not fetch any data from S3 - metadata is loaded lazily
    /// on first access.
    ///
    /// # Arguments
    ///
    /// * `s3_key` - S3 object key for the index file
    /// * `s3_client` - S3 client implementation
    /// * `config` - Index configuration
    /// * `index_id` - Unique identifier for this index
    pub fn open(
        s3_key: impl Into<String>,
        s3_client: Arc<dyn S3Client>,
        config: S3IndexConfig,
        index_id: u64,
    ) -> Self {
        let cache = S3IndexCache::new(S3IndexCacheConfig {
            max_size_bytes: config.cache_size_bytes,
        });

        Self {
            reader: Arc::new(SharedIndexReader::new(
                s3_key.into(),
                s3_client,
                cache,
            )),
            config,
            index_id,
        }
    }

    /// Create a new index with a shared cache.
    ///
    /// Use this when multiple indices should share a cache pool.
    pub fn open_with_cache(
        s3_key: impl Into<String>,
        s3_client: Arc<dyn S3Client>,
        config: S3IndexConfig,
        index_id: u64,
        cache: S3IndexCache,
    ) -> Self {
        Self {
            reader: Arc::new(SharedIndexReader::new(
                s3_key.into(),
                s3_client,
                cache,
            )),
            config,
            index_id,
        }
    }

    /// Get the S3 key for this index.
    pub fn s3_key(&self) -> &str {
        &self.reader.s3_key
    }

    /// Get the index ID.
    pub fn index_id(&self) -> u64 {
        self.index_id
    }

    /// Get the configuration.
    pub fn config(&self) -> &S3IndexConfig {
        &self.config
    }

    /// Load index metadata from S3.
    ///
    /// This is called automatically on first lookup, but can be called
    /// explicitly to pre-load metadata.
    pub async fn load(&self) -> S3IndexResult<()> {
        self.reader.ensure_loaded().await
    }

    /// Check if metadata is loaded.
    pub async fn is_loaded(&self) -> bool {
        self.reader.metadata.read().await.is_some()
    }

    /// Get the index header (loads metadata if needed).
    pub async fn header(&self) -> S3IndexResult<S3IndexHeader> {
        self.reader.ensure_loaded().await?;
        self.reader.header().await
    }

    /// Get the list of data files covered by this index.
    pub async fn files(&self) -> S3IndexResult<Vec<String>> {
        self.reader.ensure_loaded().await?;
        self.reader.files().await
    }

    /// Get the number of buckets.
    pub async fn num_buckets(&self) -> S3IndexResult<u32> {
        let header = self.header().await?;
        Ok(header.num_buckets)
    }

    /// Get the number of entries.
    pub async fn num_entries(&self) -> S3IndexResult<u64> {
        let header = self.header().await?;
        Ok(header.num_entries)
    }

    /// Find all record locations for a given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The primary key to look up (will be hashed internally)
    ///
    /// # Returns
    ///
    /// Vector of record locations matching the key. Empty if key not found.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let locations = index.find(user_id).await?;
    /// match locations.as_slice() {
    ///     [] => println!("User not found"),
    ///     [loc] => println!("Found at {}:{}", loc.file_path, loc.row_idx),
    ///     locs => println!("Found {} locations (duplicates?)", locs.len()),
    /// }
    /// ```
    pub async fn find(&self, key: u64) -> S3IndexResult<Vec<RecordLocation>> {
        self.reader.find(key).await
    }

    /// Find record locations for multiple keys.
    ///
    /// This is more efficient than calling `find` repeatedly because:
    /// - Keys mapping to the same bucket share a single S3 request
    /// - Reduces round-trip latency for batch operations
    ///
    /// # Arguments
    ///
    /// * `keys` - Slice of primary keys to look up
    ///
    /// # Returns
    ///
    /// Vector of (key, locations) pairs for each input key.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = index.find_batch(&[100, 200, 300]).await?;
    /// for (key, locations) in results {
    ///     if locations.is_empty() {
    ///         println!("Key {} not found", key);
    ///     } else {
    ///         println!("Key {} found at {} locations", key, locations.len());
    ///     }
    /// }
    /// ```
    pub async fn find_batch(&self, keys: &[u64]) -> S3IndexResult<Vec<(u64, Vec<RecordLocation>)>> {
        self.reader.find_batch(keys).await
    }

    /// Check if a key exists in the index.
    ///
    /// Equivalent to `!self.find(key).await?.is_empty()` but may be
    /// slightly more efficient in the future.
    pub async fn contains(&self, key: u64) -> S3IndexResult<bool> {
        Ok(!self.find(key).await?.is_empty())
    }

    /// Invalidate cached entries for this index.
    ///
    /// Call this if the index file has been updated externally.
    pub async fn invalidate_cache(&self) {
        let mut cache = self.reader.cache.write().await;
        cache.invalidate_index(&self.reader.s3_key);
    }

    /// Get cache statistics.
    pub async fn cache_stats(&self) -> super::cache::CacheStats {
        let cache = self.reader.cache.read().await;
        cache.stats()
    }

    /// Delete this index from S3.
    ///
    /// # Warning
    ///
    /// This permanently deletes the index file. Make sure it's no longer needed.
    pub async fn delete(self) -> S3IndexResult<()> {
        self.reader.s3_client.delete_object(&self.reader.s3_key).await
    }
}

impl std::fmt::Debug for S3GlobalIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3GlobalIndex")
            .field("s3_key", &self.reader.s3_key)
            .field("index_id", &self.index_id)
            .field("config", &self.config)
            .finish()
    }
}

// Implement PartialEq based on index_id and s3_key
impl PartialEq for S3GlobalIndex {
    fn eq(&self, other: &Self) -> bool {
        self.index_id == other.index_id && self.reader.s3_key == other.reader.s3_key
    }
}

impl Eq for S3GlobalIndex {}

// Implement Hash based on index_id
impl std::hash::Hash for S3GlobalIndex {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index_id.hash(state);
        self.reader.s3_key.hash(state);
    }
}
