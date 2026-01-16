//! LRU cache for S3 index bucket entries.
//!
//! This module provides an in-memory cache for bucket entries fetched from S3,
//! reducing the number of S3 range requests for repeated lookups.

use std::num::NonZeroUsize;

use lru::LruCache;

use super::entry::S3IndexEntry;

/// Configuration for the S3 index cache.
#[derive(Clone, Debug)]
pub struct S3IndexCacheConfig {
    /// Maximum memory size for the cache in bytes.
    pub max_size_bytes: u64,
}

impl Default for S3IndexCacheConfig {
    fn default() -> Self {
        Self {
            max_size_bytes: 100 * 1024 * 1024, // 100 MB
        }
    }
}

/// Cache key identifying a specific bucket in a specific index.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CacheKey {
    /// S3 key of the index file.
    pub index_key: String,
    /// Bucket index within the index.
    pub bucket_idx: u32,
}

impl CacheKey {
    /// Create a new cache key.
    pub fn new(index_key: impl Into<String>, bucket_idx: u32) -> Self {
        Self {
            index_key: index_key.into(),
            bucket_idx,
        }
    }

    /// Estimate memory size of this key.
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.index_key.len()
    }
}

/// Cached bucket data including entries and size information.
#[derive(Clone, Debug)]
pub struct CachedBucket {
    /// The entries in this bucket.
    pub entries: Vec<S3IndexEntry>,
    /// Estimated size of this cache entry in bytes.
    pub size_bytes: usize,
}

impl CachedBucket {
    /// Create a new cached bucket.
    pub fn new(entries: Vec<S3IndexEntry>) -> Self {
        // Estimate size: entry struct + overhead
        let size_bytes = std::mem::size_of::<Self>()
            + entries.len() * std::mem::size_of::<S3IndexEntry>()
            + entries.capacity() * std::mem::size_of::<S3IndexEntry>(); // Account for Vec capacity

        Self {
            entries,
            size_bytes,
        }
    }

    /// Check if this bucket is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// LRU cache for S3 index bucket entries.
///
/// Caches bucket entries fetched from S3 to reduce the number of range requests.
/// Uses memory-aware eviction to stay within configured limits.
///
/// # Example
///
/// ```rust,ignore
/// let mut cache = S3IndexCache::new(S3IndexCacheConfig {
///     max_size_bytes: 50 * 1024 * 1024, // 50 MB
/// });
///
/// // Check cache before fetching from S3
/// let key = CacheKey::new("indices/idx_001.bin", 623);
/// if let Some(bucket) = cache.get(&key) {
///     // Cache hit - use cached entries
///     return filter_entries(&bucket.entries, lower_hash);
/// }
///
/// // Cache miss - fetch from S3
/// let entries = fetch_bucket_from_s3(623).await?;
/// cache.insert(key, CachedBucket::new(entries.clone()));
/// ```
pub struct S3IndexCache {
    /// LRU cache mapping bucket keys to cached entries.
    cache: LruCache<CacheKey, CachedBucket>,

    /// Current total size of cached data in bytes.
    current_size_bytes: u64,

    /// Maximum allowed size in bytes.
    max_size_bytes: u64,
}

impl S3IndexCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: S3IndexCacheConfig) -> Self {
        // Estimate a reasonable max number of entries based on average bucket size
        // Assume average bucket is ~1KB, so max entries = max_size / 1KB
        let estimated_max_entries = ((config.max_size_bytes / 1024) as usize).max(100);

        Self {
            cache: LruCache::new(NonZeroUsize::new(estimated_max_entries).unwrap()),
            current_size_bytes: 0,
            max_size_bytes: config.max_size_bytes,
        }
    }

    /// Create a cache with a specific capacity (for testing).
    pub fn with_capacity(max_entries: usize, max_size_bytes: u64) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(max_entries.max(1)).unwrap()),
            current_size_bytes: 0,
            max_size_bytes,
        }
    }

    /// Get a cached bucket, returning None if not in cache.
    ///
    /// This also promotes the entry to most-recently-used.
    pub fn get(&mut self, key: &CacheKey) -> Option<&CachedBucket> {
        self.cache.get(key)
    }

    /// Check if a key exists in the cache without promoting it.
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.cache.contains(key)
    }

    /// Insert a bucket into the cache.
    ///
    /// May evict least-recently-used entries to stay within memory limits.
    pub fn insert(&mut self, key: CacheKey, bucket: CachedBucket) {
        let new_size = (bucket.size_bytes + key.size_bytes()) as u64;

        // Evict entries until we have space
        while self.current_size_bytes + new_size > self.max_size_bytes && !self.cache.is_empty() {
            if let Some((evicted_key, evicted)) = self.cache.pop_lru() {
                let evicted_size = (evicted.size_bytes + evicted_key.size_bytes()) as u64;
                self.current_size_bytes = self.current_size_bytes.saturating_sub(evicted_size);
            } else {
                break;
            }
        }

        // If the entry itself is larger than max size, don't cache it
        if new_size > self.max_size_bytes {
            return;
        }

        // Remove existing entry if present
        if let Some((old_key, old)) = self.cache.pop_entry(&key) {
            let old_size = (old.size_bytes + old_key.size_bytes()) as u64;
            self.current_size_bytes = self.current_size_bytes.saturating_sub(old_size);
        }

        self.cache.put(key, bucket);
        self.current_size_bytes += new_size;
    }

    /// Remove all entries for a specific index.
    ///
    /// Called when an index is deleted or replaced.
    pub fn invalidate_index(&mut self, index_key: &str) {
        // Collect keys to remove (can't modify while iterating)
        let keys_to_remove: Vec<CacheKey> = self
            .cache
            .iter()
            .filter(|(k, _)| k.index_key == index_key)
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            if let Some((removed_key, removed)) = self.cache.pop_entry(&key) {
                let size = (removed.size_bytes + removed_key.size_bytes()) as u64;
                self.current_size_bytes = self.current_size_bytes.saturating_sub(size);
            }
        }
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_size_bytes = 0;
    }

    /// Get current cache size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.current_size_bytes
    }

    /// Get number of cached buckets.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            num_entries: self.cache.len(),
            size_bytes: self.current_size_bytes,
            max_size_bytes: self.max_size_bytes,
        }
    }
}

/// Cache statistics for monitoring.
#[derive(Clone, Debug)]
pub struct CacheStats {
    /// Number of entries in cache.
    pub num_entries: usize,
    /// Current size in bytes.
    pub size_bytes: u64,
    /// Maximum allowed size in bytes.
    pub max_size_bytes: u64,
}

impl CacheStats {
    /// Get cache utilization as a percentage (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        if self.max_size_bytes == 0 {
            0.0
        } else {
            self.size_bytes as f64 / self.max_size_bytes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = S3IndexCache::with_capacity(100, 1024 * 1024);

        let key = CacheKey::new("index_1.bin", 42);
        let bucket = CachedBucket::new(vec![
            S3IndexEntry::new(0x1234, 0, 100),
            S3IndexEntry::new(0x5678, 1, 200),
        ]);

        cache.insert(key.clone(), bucket);

        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved.entries.len(), 2);
        assert_eq!(retrieved.entries[0].lower_hash, 0x1234);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = S3IndexCache::with_capacity(100, 1024 * 1024);

        let key = CacheKey::new("index_1.bin", 42);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_eviction() {
        // Small cache that can hold ~2 entries
        let mut cache = S3IndexCache::with_capacity(10, 200);

        let key1 = CacheKey::new("idx", 1);
        let key2 = CacheKey::new("idx", 2);
        let key3 = CacheKey::new("idx", 3);

        // Insert entries that will exceed cache size
        cache.insert(key1.clone(), CachedBucket::new(vec![S3IndexEntry::new(1, 0, 0)]));
        cache.insert(key2.clone(), CachedBucket::new(vec![S3IndexEntry::new(2, 0, 0)]));
        cache.insert(key3.clone(), CachedBucket::new(vec![S3IndexEntry::new(3, 0, 0)]));

        // At least one entry should be evicted, most recent should remain
        assert!(cache.get(&key3).is_some());
    }

    #[test]
    fn test_cache_invalidate_index() {
        let mut cache = S3IndexCache::with_capacity(100, 1024 * 1024);

        // Insert entries for two different indices
        cache.insert(
            CacheKey::new("index_1.bin", 1),
            CachedBucket::new(vec![S3IndexEntry::new(1, 0, 0)]),
        );
        cache.insert(
            CacheKey::new("index_1.bin", 2),
            CachedBucket::new(vec![S3IndexEntry::new(2, 0, 0)]),
        );
        cache.insert(
            CacheKey::new("index_2.bin", 1),
            CachedBucket::new(vec![S3IndexEntry::new(3, 0, 0)]),
        );

        assert_eq!(cache.len(), 3);

        // Invalidate index_1
        cache.invalidate_index("index_1.bin");

        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&CacheKey::new("index_2.bin", 1)));
        assert!(!cache.contains(&CacheKey::new("index_1.bin", 1)));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = S3IndexCache::with_capacity(100, 1024 * 1024);

        cache.insert(
            CacheKey::new("idx", 1),
            CachedBucket::new(vec![S3IndexEntry::new(1, 0, 0)]),
        );
        cache.insert(
            CacheKey::new("idx", 2),
            CachedBucket::new(vec![S3IndexEntry::new(2, 0, 0)]),
        );

        assert_eq!(cache.len(), 2);
        assert!(cache.size_bytes() > 0);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.size_bytes(), 0);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = S3IndexCache::with_capacity(100, 1000);

        cache.insert(
            CacheKey::new("idx", 1),
            CachedBucket::new(vec![S3IndexEntry::new(1, 0, 0)]),
        );

        let stats = cache.stats();
        assert_eq!(stats.num_entries, 1);
        assert!(stats.size_bytes > 0);
        assert_eq!(stats.max_size_bytes, 1000);
        assert!(stats.utilization() > 0.0);
        assert!(stats.utilization() <= 1.0);
    }
}
