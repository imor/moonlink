//! Configuration for S3 index.
//!
//! This module defines configuration structures for S3-based indices.

use serde::{Deserialize, Serialize};

/// Default number of buckets for the index (64K buckets).
pub const DEFAULT_NUM_BUCKETS: u32 = 65536;

/// Default cache size (100 MB).
pub const DEFAULT_CACHE_SIZE_BYTES: u64 = 100 * 1024 * 1024;

/// Configuration for S3 index behavior.
///
/// # Example
///
/// ```rust,ignore
/// let config = S3IndexConfig {
///     cache_size_bytes: 256 * 1024 * 1024, // 256 MB cache
///     num_buckets: 65536,                   // 64K buckets
///     s3_bucket: "my-data-bucket".into(),
///     s3_prefix: "indices/table_123".into(),
/// };
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct S3IndexConfig {
    /// Maximum memory for entry cache (bytes).
    ///
    /// This controls how many bucket entries are kept in memory.
    /// Larger cache = fewer S3 requests = better performance.
    ///
    /// Default: 100 MB
    pub cache_size_bytes: u64,

    /// Number of buckets in the hash index.
    ///
    /// More buckets = smaller bucket size = faster lookups.
    /// Should be a power of 2 for optimal hash distribution.
    ///
    /// Default: 65536 (2^16)
    pub num_buckets: u32,

    /// S3 bucket name.
    pub s3_bucket: String,

    /// Prefix for index files in S3.
    ///
    /// Index files will be stored at: `{s3_prefix}/index_{id}.bin`
    ///
    /// Example: `"indices/table_123"` → `"indices/table_123/index_001.bin"`
    pub s3_prefix: String,
}

impl S3IndexConfig {
    /// Create a new configuration with defaults.
    ///
    /// # Arguments
    ///
    /// * `s3_bucket` - S3 bucket name
    /// * `s3_prefix` - Prefix for index files
    pub fn new(s3_bucket: impl Into<String>, s3_prefix: impl Into<String>) -> Self {
        Self {
            cache_size_bytes: DEFAULT_CACHE_SIZE_BYTES,
            num_buckets: DEFAULT_NUM_BUCKETS,
            s3_bucket: s3_bucket.into(),
            s3_prefix: s3_prefix.into(),
        }
    }

    /// Set the cache size.
    pub fn with_cache_size(mut self, cache_size_bytes: u64) -> Self {
        self.cache_size_bytes = cache_size_bytes;
        self
    }

    /// Set the number of buckets.
    pub fn with_num_buckets(mut self, num_buckets: u32) -> Self {
        self.num_buckets = num_buckets;
        self
    }

    /// Generate S3 key for an index file.
    ///
    /// # Arguments
    ///
    /// * `index_id` - Unique identifier for the index
    ///
    /// # Returns
    ///
    /// S3 key in format: `{prefix}/index_{id}.bin`
    pub fn index_key(&self, index_id: u64) -> String {
        format!("{}/index_{}.bin", self.s3_prefix, index_id)
    }
}

impl Default for S3IndexConfig {
    fn default() -> Self {
        Self {
            cache_size_bytes: DEFAULT_CACHE_SIZE_BYTES,
            num_buckets: DEFAULT_NUM_BUCKETS,
            s3_bucket: String::new(),
            s3_prefix: String::new(),
        }
    }
}
