//! # Index Cache Utilities
//!
//! This module provides utilities for integrating file indices with the object storage cache.
//!
//! ## Overview
//!
//! When index files are created or loaded from disk, they need to be imported into the
//! cache system to ensure:
//! 1. **Fast access** via memory-mapped I/O
//! 2. **Cache management** - evicting old indices when space is limited
//! 3. **Reference counting** - keeping track of which indices are actively in use
//!
//! ## Cache Integration Flow
//!
//! ```text
//! ┌────────────────────────────────────┐
//! │  1. Create Index File on Disk      │
//! │     (persisted_bucket_hash_map.rs) │
//! └─────────────┬──────────────────────┘
//!               │
//!               ▼
//! ┌────────────────────────────────────┐
//! │  2. Import to Cache                │
//! │     (import_file_index_to_cache)   │
//! │  - Creates cache entry             │
//! │  - Gets non-evictable handle       │
//! │  - May evict old indices           │
//! └─────────────┬──────────────────────┘
//!               │
//!               ▼
//! ┌────────────────────────────────────┐
//! │  3. Use Index (via cache handle)   │
//! │  - Lookups read from cache         │
//! │  - Handle keeps it non-evictable   │
//! └─────────────┬──────────────────────┘
//!               │
//!               ▼
//! ┌────────────────────────────────────┐
//! │  4. Unreference and Delete         │
//! │     (when index no longer needed)  │
//! │  - Drop handle → evictable         │
//! │  - Eventually removed from cache   │
//! └────────────────────────────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! // After building a new file index
//! let mut file_index = builder.build_from_flush(hash_entries, file_id).await?;
//!
//! // Import all index blocks into cache
//! let evicted_files = import_file_index_to_cache(
//!     &mut file_index,
//!     object_storage_cache.clone(),
//!     table_id,
//! ).await;
//!
//! // Delete any files that were evicted from cache
//! for file_path in evicted_files {
//!     tokio::fs::remove_file(&file_path).await?;
//! }
//!
//! // Now the index can be used efficiently
//! // Its cache handle keeps it loaded in memory
//!
//! // Later, when index is replaced by merged version
//! let evicted_files = unreference_and_delete_file_index_from_cache(
//!     &mut file_index
//! ).await;
//! ```
//!
//! ## Cache Eviction
//!
//! The cache has limited capacity. When importing new indices:
//! - **Non-evictable** indices (with handles) stay in cache
//! - **Evictable** indices (handles dropped) can be removed
//! - Least Recently Used (LRU) eviction policy
//!
//! Files evicted from cache are returned so they can be deleted from disk.

use std::sync::Arc;

use crate::storage::cache::object_storage::base_cache::{CacheEntry, CacheTrait, FileMetadata};
use crate::storage::index::persisted_bucket_hash_map::GlobalIndex;
use crate::storage::storage_utils::{TableId, TableUniqueFileId};

/// Import a file index into the object storage cache.
///
/// This function registers each index block with the cache system and obtains
/// non-evictable handles to keep them loaded in memory for fast access.
///
/// # Arguments
///
/// * `file_index` - The file index to import (will be modified to store cache handles)
/// * `object_storage_cache` - The cache to import into
/// * `table_id` - Table identifier for cache key construction
///
/// # Returns
///
/// A vector of file paths that were evicted from cache to make room. These files
/// should typically be deleted from disk since they're no longer in cache.
///
/// # How It Works
///
/// For each index block in the file index:
///
/// 1. **Create cache entry** with file path and metadata (size)
/// 2. **Generate unique key** combining table_id and file_id
/// 3. **Import to cache** - may trigger eviction of old entries
/// 4. **Store handle** in index block for future access
///
/// # Example
///
/// ```rust,ignore
/// // Build a new file index
/// let mut file_index = GlobalIndexBuilder::new()
///     .set_files(vec![data_file])
///     .build_from_flush(hash_entries, file_id).await?;
///
/// // Import to cache
/// let evicted = import_file_index_to_cache(
///     &mut file_index,
///     cache.clone(),
///     TableId(42),
/// ).await;
///
/// println!("Imported index, evicted {} files", evicted.len());
/// // evicted might contain: ["old_index_123.bin", "old_index_456.bin"]
///
/// // Clean up evicted files
/// for path in evicted {
///     tokio::fs::remove_file(path).await?;
/// }
/// ```
///
/// # Cache State Changes
///
/// **Before:**
/// ```text
/// Cache: [index_100 (evictable), index_101 (non-evictable)]
/// ```
///
/// **After importing index_102 (2 blocks):**
/// ```text
/// Cache: [index_101 (non-evictable), index_102_block0 (non-evictable),
///         index_102_block1 (non-evictable)]
/// Evicted: ["index_100.bin"]  // Returned to caller
/// ```
///
/// # Panics
///
/// This function does not panic under normal circumstances.
pub async fn import_file_index_to_cache(
    file_index: &mut GlobalIndex,
    object_storage_cache: Arc<dyn CacheTrait>,
    table_id: TableId,
) -> Vec<String> {
    // Aggregate evicted files to delete.
    let mut evicted_files_to_delete = vec![];

    for cur_index_block in file_index.index_blocks.iter_mut() {
        let table_unique_file_id = TableUniqueFileId {
            table_id,
            file_id: cur_index_block.index_file.file_id(),
        };
        let cache_entry = CacheEntry {
            cache_filepath: cur_index_block.index_file.file_path().clone(),
            file_metadata: FileMetadata {
                file_size: cur_index_block.file_size,
            },
        };
        let (cache_handle, cur_evicted_files) = object_storage_cache
            .import_cache_entry(table_unique_file_id, cache_entry)
            .await;
        evicted_files_to_delete.extend(cur_evicted_files);
        cur_index_block.cache_handle = Some(cache_handle);
    }

    evicted_files_to_delete
}

/// Import multiple file indices into the cache at once.
///
/// This is a convenience function that imports several file indices in sequence,
/// aggregating all evicted files from each import operation.
///
/// # Arguments
///
/// * `file_indices` - Slice of file indices to import (each will be modified)
/// * `object_storage_cache` - The cache to import into
/// * `table_id` - Table identifier for all indices
///
/// # Returns
///
/// Combined list of all file paths evicted during any of the imports.
///
/// # Example
///
/// ```rust,ignore
/// // After index merge operation creates 3 new indices
/// let mut new_indices = vec![index1, index2, index3];
///
/// // Import all at once
/// let evicted = import_file_indices_to_cache(
///     &mut new_indices,
///     cache.clone(),
///     TableId(42),
/// ).await;
///
/// println!("Imported {} indices, evicted {} files",
///          new_indices.len(), evicted.len());
///
/// // Clean up all evicted files
/// for path in evicted {
///     tokio::fs::remove_file(path).await?;
/// }
/// ```
///
/// # Execution Flow
///
/// ```text
/// Import index1 → evicted: ["a.bin"]
///   ↓
/// Import index2 → evicted: ["b.bin", "c.bin"]
///   ↓  
/// Import index3 → evicted: []
///   ↓
/// Return: ["a.bin", "b.bin", "c.bin"]
/// ```
///
/// # Performance Note
///
/// Imports are done sequentially, not in parallel, to avoid race conditions
/// in cache management.
pub async fn import_file_indices_to_cache(
    file_indices: &mut [GlobalIndex],
    object_storage_cache: Arc<dyn CacheTrait>,
    table_id: TableId,
) -> Vec<String> {
    // Aggregate evicted files to delete.
    let mut evicted_files_to_delete = vec![];

    for cur_file_index in file_indices.iter_mut() {
        let cur_evicted_files =
            import_file_index_to_cache(cur_file_index, object_storage_cache.clone(), table_id)
                .await;
        evicted_files_to_delete.extend(cur_evicted_files);
    }

    evicted_files_to_delete
}

/// Remove a file index from cache and mark it for deletion.
///
/// This function releases all cache handles associated with an index, making its
/// blocks eligible for eviction. This is typically called when:
/// - An index has been merged into a larger index
/// - An index is no longer needed (e.g., after table drop)
/// - Rolling back a failed operation
///
/// # Arguments
///
/// * `file_index` - The index to remove (must have been previously imported)
///
/// # Returns
///
/// Vector of file paths that became evicted as a result. The caller should
/// delete these files from disk.
///
/// # Precondition
///
/// All index blocks MUST have cache handles (i.e., must have been previously
/// imported via `import_file_index_to_cache`). Otherwise, this function will panic.
///
/// # Example
///
/// ```rust,ignore
/// // After merging several small indices into one large index
/// let mut old_indices = vec![small_index1, small_index2, small_index3];
/// let new_merged_index = merge_indices(&old_indices).await?;
///
/// // Import the new index
/// import_file_index_to_cache(&mut new_merged_index, cache.clone(), table_id).await;
///
/// // Remove old indices from cache
/// let mut all_evicted = vec![];
/// for old_index in &mut old_indices {
///     let evicted = unreference_and_delete_file_index_from_cache(old_index).await;
///     all_evicted.extend(evicted);
/// }
///
/// // Delete evicted files
/// for path in all_evicted {
///     tokio::fs::remove_file(path).await?;
/// }
/// ```
///
/// # What Happens
///
/// For each index block:
/// 1. **Unreference handle** - decrements reference count
/// 2. **If count reaches 0** - move from non-evictable to evictable cache
/// 3. **If cache is full** - evict immediately and return file path
/// 4. **Clear handle** - set to None in the index block
///
/// # Cache State Example
///
/// **Before:**
/// ```text
/// Non-evictable: [index_100 (refcount=1), index_101 (refcount=2)]
/// Evictable: [index_99]
/// ```
///
/// **After unreferencing index_100:**
/// ```text
/// Non-evictable: [index_101 (refcount=2)]
/// Evictable: [index_99, index_100]  // or evicted if cache full
/// Returned: ["index_100.bin"]  // if evicted
/// ```
///
/// # Panics
///
/// Panics if any index block does not have a cache handle, indicating it
/// was never imported or has already been removed.
pub async fn unreference_and_delete_file_index_from_cache(
    file_index: &mut GlobalIndex,
) -> Vec<String> {
    let mut evicted_files_to_delete = vec![];
    for cur_index_block in file_index.index_blocks.iter_mut() {
        assert!(cur_index_block.cache_handle.is_some());
        let cur_evicted_files = cur_index_block
            .cache_handle
            .as_mut()
            .unwrap()
            .unreference_and_delete()
            .await;
        evicted_files_to_delete.extend(cur_evicted_files);
        cur_index_block.cache_handle = None;
    }
    evicted_files_to_delete
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_data_file;
    use crate::storage::index::persisted_bucket_hash_map::GlobalIndexBuilder;
    use crate::ObjectStorageCache;

    #[tokio::test]
    async fn test_import_index_to_cache() {
        let temp_dir = tempfile::tempdir().unwrap();
        let object_storage_cache = ObjectStorageCache::default_for_test(&temp_dir);
        // Create first file index.
        let mut builder = GlobalIndexBuilder::new();
        builder
            .set_files(vec![create_data_file(
                /*file_id=*/ 0,
                "a.parquet".to_string(),
            )])
            .set_directory(tempfile::tempdir().unwrap().keep());
        let file_index_1 = builder
            .build_from_flush(/*hash_entries=*/ vec![(1, 0, 0)], /*file_id=*/ 1)
            .await
            .unwrap();

        // Create second file index.
        let mut builder = GlobalIndexBuilder::new();
        builder
            .set_files(vec![create_data_file(
                /*file_id=*/ 2,
                "b.parquet".to_string(),
            )])
            .set_directory(tempfile::tempdir().unwrap().keep());
        let file_index_2 = builder
            .build_from_flush(/*hash_entries=*/ vec![(2, 0, 0)], /*file_id=*/ 3)
            .await
            .unwrap();

        let mut index_block_files = vec![
            file_index_1.index_blocks[0].index_file.file_path().clone(),
            file_index_2.index_blocks[0].index_file.file_path().clone(),
        ];
        index_block_files.sort();

        let mut file_indices = vec![file_index_1, file_index_2];
        import_file_indices_to_cache(
            &mut file_indices,
            Arc::new(object_storage_cache.clone()),
            TableId(0),
        )
        .await;

        // Check both file indices are pinned in cache.
        assert_eq!(
            object_storage_cache
                .cache
                .read()
                .await
                .non_evictable_cache
                .len(),
            2
        );
        assert_eq!(
            object_storage_cache
                .cache
                .read()
                .await
                .evictable_cache
                .len(),
            0
        );
        assert_eq!(
            object_storage_cache
                .cache
                .read()
                .await
                .evicted_entries
                .len(),
            0
        );

        // Check cache handle is assigned to the file indice.
        assert!(file_indices[0].index_blocks[0].cache_handle.is_some());
        assert!(file_indices[1].index_blocks[0].cache_handle.is_some());

        // Unreference and delete all file indices.
        let mut evicted_files_to_delete = vec![];
        for cur_file_index in file_indices.iter_mut() {
            let cur_evicted_files =
                unreference_and_delete_file_index_from_cache(cur_file_index).await;
            evicted_files_to_delete.extend(cur_evicted_files);
        }
        evicted_files_to_delete.sort();
        assert_eq!(evicted_files_to_delete, index_block_files);

        // Check both file indices are pinned in cache.
        assert_eq!(
            object_storage_cache
                .cache
                .read()
                .await
                .non_evictable_cache
                .len(),
            0
        );
        assert_eq!(
            object_storage_cache
                .cache
                .read()
                .await
                .evictable_cache
                .len(),
            0
        );
        assert_eq!(
            object_storage_cache
                .cache
                .read()
                .await
                .evicted_entries
                .len(),
            0
        );

        // Check cache handle is assigned to the file indice.
        assert!(file_indices[0].index_blocks[0].cache_handle.is_none());
        assert!(file_indices[1].index_blocks[0].cache_handle.is_none());
    }
}
