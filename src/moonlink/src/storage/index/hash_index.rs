//! # Hash Index Implementation
//!
//! This module implements the core indexing operations for `MooncakeIndex`:
//! - Creating new indices
//! - Inserting and deleting memory indices  
//! - Finding records across memory and disk
//!
//! ## Key Operations
//!
//! ### 1. Creating an Index
//!
//! ```rust,ignore
//! let index = MooncakeIndex::new();
//! ```
//!
//! Creates an empty index with no memory or file indices.
//!
//! ### 2. Adding Memory Indices
//!
//! ```rust,ignore
//! // When a new memory batch is created
//! let mem_index = Arc::new(MemIndex::new(IdentityProp::SinglePrimitiveKey(0)));
//! index.insert_memory_index(mem_index);
//! ```
//!
//! ### 3. Finding Records
//!
//! ```rust,ignore
//! let deletion_record = RawDeletionRecord {
//!     lookup_key: 12345,
//!     row_identity: None,
//!     pos: None,
//!     lsn: 100,
//!     delete_if_exists: false,
//! };
//!
//! // Searches both memory and disk
//! let locations = index.find_record(&deletion_record).await;
//! // Returns: vec![RecordLocation::MemoryBatch(2, 15),
//! //               RecordLocation::DiskFile(3, 892)]
//! ```
//!
//! ### 4. Batch Finding
//!
//! ```rust,ignore
//! // More efficient when looking up many records
//! let records = vec![record1, record2, record3];
//! let results = index.find_records(&records).await;
//! // Returns: vec![(key1, location1), (key2, location2), ...]
//! ```
//!
//! ## Search Strategy
//!
//! When finding a record, the index searches both tiers:
//!
//! ```text
//! Query: Find lookup_key = 12345
//!   │
//!   ├── Search In-Memory Indices (uses lookup_key directly)
//!   │   ├── Check mem_index_1 → Not found
//!   │   ├── Check mem_index_2 → Found at MemoryBatch(2, 15)
//!   │   └── Check mem_index_3 → Not found
//!   │
//!   └── Search File Indices (hashes the lookup_key)
//!       ├── Hash: splitmix64(12345) → bucket + lower bits
//!       ├── Check file_index_1 → Not found
//!       └── Check file_index_2 → Found at DiskFile(3, 892)
//!
//! Result: [MemoryBatch(2, 15), DiskFile(3, 892)]
//! ```
//!
//! ## Deduplication
//!
//! - **In-Memory**: Uses `HashSet` to deduplicate (same key may exist in multiple batches)
//! - **File Indices**: No deduplication needed (each index covers different data)

use crate::storage::index::persisted_bucket_hash_map::splitmix64;
use crate::storage::index::*;
use crate::storage::storage_utils::{RawDeletionRecord, RecordLocation};
use std::collections::HashSet;
use std::sync::Arc;

impl MooncakeIndex {
    /// Create a new, empty index.
    ///
    /// Initializes an index with no memory indices and no file indices.
    /// Indices are added later as memory batches are created and data is flushed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let index = MooncakeIndex::new();
    /// assert!(index.in_memory_index.is_empty());
    /// assert!(index.file_indices.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            in_memory_index: HashSet::new(),
            file_indices: Vec::new(),
        }
    }

    /// Add a memory index to the active index set.
    ///
    /// Called when a new memory batch is created. The memory index tracks all
    /// records in that specific batch, enabling fast lookups before flush.
    ///
    /// # Arguments
    ///
    /// * `mem_index` - Arc-wrapped memory index for a single memory batch
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create a new memory batch
    /// let batch_id = 42;
    /// let mem_index = Arc::new(MemIndex::new(IdentityProp::SinglePrimitiveKey(0)));
    ///
    /// // Populate it with records
    /// mem_index.insert(key=100, identity=None, RecordLocation::MemoryBatch(42, 0));
    /// mem_index.insert(key=101, identity=None, RecordLocation::MemoryBatch(42, 1));
    ///
    /// // Add to main index
    /// index.insert_memory_index(mem_index);
    ///
    /// // Now lookups for key=100 or key=101 will find them
    /// ```
    ///
    /// # Index Lifecycle
    ///
    /// ```text
    /// 1. Create memory batch → create MemIndex
    /// 2. Insert records → populate MemIndex  
    /// 3. insert_memory_index() → make searchable
    /// 4. Flush to disk → delete_memory_index()
    /// 5. Build file index → insert_file_index()
    /// ```
    pub fn insert_memory_index(&mut self, mem_index: Arc<MemIndex>) {
        self.in_memory_index.insert(IndexPtr(mem_index));
    }

    /// Remove a memory index from the active set.
    ///
    /// Called when a memory batch has been flushed to disk. The memory index is
    /// no longer needed since the data is now indexed via a file index.
    ///
    /// # Arguments
    ///
    /// * `mem_index` - The specific memory index to remove (matched by Arc pointer)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Flush memory batch to disk
    /// let parquet_file = flush_batch_to_disk(&batch).await?;
    ///
    /// // Build file index for the new parquet file
    /// let file_index = build_index_for_file(&parquet_file).await?;
    /// index.insert_file_index(file_index);
    ///
    /// // Now we can remove the memory index
    /// index.delete_memory_index(&mem_index);
    ///
    /// // Searches will now use file index instead
    /// ```
    ///
    /// # Why Arc Pointer Matching?
    ///
    /// Uses pointer identity (via `IndexPtr`) to ensure we remove the exact
    /// memory index for this batch, even if another batch has identical contents.
    pub fn delete_memory_index(&mut self, mem_index: &Arc<MemIndex>) {
        self.in_memory_index.remove(&IndexPtr(mem_index.clone()));
    }

    /// Add a file index to the collection.
    ///
    /// Called after data has been flushed to Parquet files and a persistent index
    /// has been built. The file index enables lookups in the flushed data.
    ///
    /// # Arguments
    ///
    /// * `file_index` - A GlobalIndex covering one or more Parquet files
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // After flushing memory batches to disk
    /// let parquet_files = vec![file1, file2, file3];
    ///
    /// // Build an index covering these files
    /// let file_index = GlobalIndexBuilder::new()
    ///     .set_files(parquet_files)
    ///     .build_from_flush(hash_entries, file_id).await?;
    ///
    /// // Add to main index
    /// index.insert_file_index(file_index);
    ///
    /// // Now lookups will search these parquet files too
    /// ```
    ///
    /// # File Index Accumulation
    ///
    /// File indices accumulate over time:
    /// ```text
    /// Flush 1: [file1, file2] → index_1
    /// Flush 2: [file3, file4] → index_2  
    /// Flush 3: [file5] → index_3
    ///
    /// file_indices = [index_1, index_2, index_3]
    /// ```
    ///
    /// Eventually these are merged to reduce overhead (see index_merge_config).
    pub fn insert_file_index(&mut self, file_index: FileIndex) {
        self.file_indices.push(file_index);
    }
}

impl MooncakeIndex {
    /// Find all locations of a single record.
    ///
    /// Searches both in-memory and file indices for the given key, returning all
    /// matching locations. Multiple locations can exist during compaction or when
    /// using FullRow identity (which allows duplicates).
    ///
    /// # Arguments
    ///
    /// * `raw_record` - The deletion record containing:
    ///   - `lookup_key`: The primary key hash to search for
    ///   - `row_identity`: For composite keys, the full key columns (optional)
    ///   - Other metadata (lsn, pos, etc.)
    ///
    /// # Returns
    ///
    /// Vector of all matching record locations. Empty if not found.
    ///
    /// # Example: Single Primitive Key
    ///
    /// ```rust,ignore
    /// let record = RawDeletionRecord {
    ///     lookup_key: 12345,  // User ID
    ///     row_identity: None,  // Not needed for single primitive key
    ///     pos: None,
    ///     lsn: 100,
    ///     delete_if_exists: false,
    /// };
    ///
    /// let locations = index.find_record(&record).await;
    /// // Possible results:
    /// // [] - not found
    /// // [MemoryBatch(2, 15)] - found in memory
    /// // [DiskFile(3, 892)] - found on disk
    /// // [MemoryBatch(2, 15), DiskFile(3, 892)] - found in both (during compaction)
    /// ```
    ///
    /// # Example: Composite Key
    ///
    /// ```rust,ignore
    /// let key_columns = MoonlinkRow::new(vec![
    ///     RowValue::Int32(100),  // order_id
    ///     RowValue::Int32(5),    // item_id
    /// ]);
    ///
    /// let record = RawDeletionRecord {
    ///     lookup_key: hash_of_key(&key_columns),
    ///     row_identity: Some(key_columns),  // Need full identity for verification
    ///     pos: None,
    ///     lsn: 200,
    ///     delete_if_exists: false,
    /// };
    ///
    /// let locations = index.find_record(&record).await;
    /// ```
    ///
    /// # Search Process
    ///
    /// 1. **Memory Search**:
    ///    - Iterate through all memory indices
    ///    - For each, call `mem_index.find_record()`
    ///    - Collect all matches
    ///
    /// 2. **File Search**:
    ///    - Hash the key: `splitmix64(lookup_key)`
    ///    - For each file index:
    ///      - Determine bucket from upper hash bits
    ///      - Read bucket entries from mmap'd file
    ///      - Match on lower hash bits
    ///      - Return (file_id, row_idx) pairs
    ///
    /// 3. **Combine Results**: Merge results from both tiers
    ///
    /// # Performance
    ///
    /// - **Memory lookup**: O(1) hash table lookup per memory index
    /// - **File lookup**: O(1) bucket lookup + O(k) entry scan where k = entries in bucket
    /// - **Overall**: O(m + n*k) where m = memory indices, n = file indices, k = avg bucket size
    pub async fn find_record(&self, raw_record: &RawDeletionRecord) -> Vec<RecordLocation> {
        let mut res: Vec<RecordLocation> = Vec::new();

        // Check in-memory indices
        for index in self.in_memory_index.iter() {
            res.extend(index.0.find_record(raw_record));
        }

        let value_and_hashes = vec![(raw_record.lookup_key, splitmix64(raw_record.lookup_key))];

        // Check file indices
        for file_index_meta in &self.file_indices {
            let locations = file_index_meta.search_values(&value_and_hashes).await;
            res.extend(locations.into_iter().map(|(_, location)| location));
        }
        res
    }

    /// Find locations for multiple records in a single batch.
    ///
    /// More efficient than calling `find_record()` repeatedly because:
    /// 1. Deduplicates lookup keys before querying file indices
    /// 2. Amortizes the cost of hashing and file I/O
    ///
    /// # Arguments
    ///
    /// * `raw_records` - Slice of deletion records to look up
    ///
    /// # Returns
    ///
    /// Vector of (key, location) pairs for all found records.
    /// Each input record may produce 0, 1, or multiple results.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let records = vec![
    ///     RawDeletionRecord { lookup_key: 100, ... },
    ///     RawDeletionRecord { lookup_key: 200, ... },
    ///     RawDeletionRecord { lookup_key: 300, ... },
    /// ];
    ///
    /// let results = index.find_records(&records).await;
    /// // Results might be:
    /// // [
    /// //   (100, MemoryBatch(1, 5)),
    /// //   (200, DiskFile(2, 42)),
    /// //   (200, DiskFile(3, 15)),  // duplicate key, different file
    /// //   (300, MemoryBatch(1, 8)),
    /// // ]
    /// // Note: lookup_key=100 might not be in results if not found
    /// ```
    ///
    /// # Deduplication Strategy
    ///
    /// ## In-Memory Indices
    ///
    /// Results are deduplicated using a `HashSet` because:
    /// - Same key might exist in multiple memory batches during compaction
    /// - Each batch might have different row_identity for the same key
    /// - We need all unique (key, location) pairs
    ///
    /// ```text
    /// Memory Batch 1: key=100 at row 5 with identity [1, 2]
    /// Memory Batch 2: key=100 at row 3 with identity [1, 2] (duplicate)
    /// Memory Batch 3: key=100 at row 8 with identity [1, 3] (different identity!)
    ///
    /// Without dedup: [(100, MB1-5), (100, MB2-3), (100, MB3-8)]
    /// With dedup: [(100, MB1-5), (100, MB3-8)]  // MB2 is duplicate of MB1
    /// ```
    ///
    /// ## File Indices
    ///
    /// Input keys are deduplicated before querying:
    /// ```rust,ignore
    /// // Input: [100, 200, 100, 300, 200]
    /// // Deduped: [100, 200, 300]  // Only query each unique key once
    /// ```
    ///
    /// File results don't need deduplication because each file index has unique entries.
    ///
    /// # Performance Optimization
    ///
    /// **Single lookups** (calling find_record 3 times):
    /// ```text
    /// Record 1: Hash → Check all file indices
    /// Record 2: Hash → Check all file indices  
    /// Record 3: Hash → Check all file indices
    /// Total: 3 hash operations, 3 * n file index scans
    /// ```
    ///
    /// **Batch lookup** (calling find_records once):
    /// ```text
    /// All records: Hash all → Check each file index once with all hashes
    /// Total: 3 hash operations, 1 * n file index scans
    /// ```
    ///
    /// Speedup: ~3x for file lookups when n file indices exist.
    pub async fn find_records(
        &self,
        raw_records: &[RawDeletionRecord],
    ) -> Vec<(u64, RecordLocation)> {
        let mut res: Vec<(u64, RecordLocation)> = Vec::new();
        // In memory index may produce duplicate results,
        // since we can't blindly input by key,
        // as records with same key may have different row_identity,
        // and we do use row_identity in lookup.
        // Dedup the result instead.
        let mut in_memory_res = HashSet::new();
        for index in self.in_memory_index.iter() {
            for record in raw_records {
                in_memory_res.extend(
                    index
                        .0
                        .find_record(record)
                        .into_iter()
                        .map(|location| (record.lookup_key, location)),
                );
            }
        }
        res.extend(in_memory_res.into_iter());
        if self.file_indices.is_empty() {
            return res;
        }
        // For file index, we can dedup input by key.
        let value_and_hashes = GlobalIndex::prepare_hashes_for_lookup(
            raw_records.iter().map(|record| record.lookup_key),
        );
        // Check file indices
        for file_index_meta in &self.file_indices {
            let locations = file_index_meta.search_values(&value_and_hashes).await;
            res.extend(locations);
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::row::IdentityProp;
    #[tokio::test]
    async fn test_in_memory_index_basic() {
        let mut index = MooncakeIndex::new();

        let identity = IdentityProp::SinglePrimitiveKey(0);
        // Insert memory records as a batch
        let mut mem_index = MemIndex::new(identity);
        mem_index.insert(1, None, RecordLocation::MemoryBatch(0, 5));
        mem_index.insert(2, None, RecordLocation::MemoryBatch(0, 10));
        mem_index.insert(3, None, RecordLocation::MemoryBatch(1, 3));
        index.insert_memory_index(Arc::new(mem_index));

        let record = RawDeletionRecord {
            lookup_key: 1,
            row_identity: None,
            pos: None,
            lsn: 1,
            delete_if_exists: false,
        };

        // Test the Index trait implementation
        let trait_locations = index.find_record(&record).await;
        assert_eq!(trait_locations.len(), 1);
    }
}
