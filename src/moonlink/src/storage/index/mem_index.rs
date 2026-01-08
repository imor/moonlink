//! # In-Memory Index Implementation
//!
//! This module implements the in-memory portion of Moonlink's indexing system,
//! providing fast lookups for records that haven't been flushed to disk yet.
//!
//! ## Overview
//!
//! Each memory batch (collection of recently written rows) has its own `MemIndex`
//! that tracks where each record is located within that batch.
//!
//! ## Index Variants
//!
//! The `MemIndex` enum has four variants, chosen based on table identity properties:
//!
//! ### 1. SinglePrimitive
//!
//! **Used for**: Tables with a single primitive column as primary key
//!
//! **Example**: Users table with `user_id INT PRIMARY KEY`
//!
//! ```text
//! Memory Batch:
//! Row 0: user_id=100, name="Alice", email="alice@example.com"
//! Row 1: user_id=200, name="Bob", email="bob@example.com"
//! Row 2: user_id=150, name="Carol", email="carol@example.com"
//!
//! MemIndex::SinglePrimitive:
//! 100 → RecordLocation::MemoryBatch(batch_id=5, row=0)
//! 200 → RecordLocation::MemoryBatch(batch_id=5, row=1)
//! 150 → RecordLocation::MemoryBatch(batch_id=5, row=2)
//! ```
//!
//! ### 2. Key (Composite Keys)
//!
//! **Used for**: Tables with multiple columns forming the primary key
//!
//! **Example**: Order items with `PRIMARY KEY (order_id, item_id)`
//!
//! ```text
//! Memory Batch:
//! Row 0: order_id=1000, item_id=5, quantity=3
//! Row 1: order_id=1000, item_id=7, quantity=1
//! Row 2: order_id=2000, item_id=5, quantity=2
//!
//! MemIndex::Key:
//! hash([1000, 5]) → {
//!     identity: [1000, 5],
//!     location: MemoryBatch(batch_id=3, row=0)
//! }
//! hash([1000, 7]) → {
//!     identity: [1000, 7],
//!     location: MemoryBatch(batch_id=3, row=1)
//! }
//! hash([2000, 5]) → {
//!     identity: [2000, 5],
//!     location: MemoryBatch(batch_id=3, row=2)
//! }
//! ```
//!
//! **Why store identity?** Hash collisions! Two different keys might hash to the
//! same value, so we need to verify the full identity matches.
//!
//! ### 3. FullRow
//!
//! **Used for**: Tables without a primary key (REPLICA IDENTITY FULL)
//!
//! **Example**: Event log where identical events can occur
//!
//! ```text
//! Memory Batch:
//! Row 0: timestamp="2024-01-01", event="login", user_id=100
//! Row 1: timestamp="2024-01-01", event="login", user_id=100  (identical row)
//! Row 2: timestamp="2024-01-02", event="logout", user_id=100
//!
//! MemIndex::FullRow (MultiMap - same hash can map to multiple locations):
//! hash(row0) = hash(row1) → [MemoryBatch(batch_id=7, row=0), MemoryBatch(batch_id=7, row=1)]
//! hash(row2) → [MemoryBatch(batch_id=7, row=2)]
//! ```
//!
//! ### 4. None (Append-Only)
//!
//! **Used for**: Append-only tables with no primary key
//!
//! **Example**: Time-series sensor data
//!
//! ```text
//! No index needed - records are never looked up or deleted.
//! All operations (insert, find, delete) will panic if called.
//! ```
//!
//! ## Key Operations
//!
//! ### Creating an Index
//!
//! ```rust,ignore
//! // Choose variant based on table schema
//! let index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
//! ```
//!
//! ### Inserting Records
//!
//! ```rust,ignore
//! // Single primitive key
//! index.insert(
//!     key=12345,
//!     identity_for_key=None,  // Not needed for single primitive
//!     RecordLocation::MemoryBatch(batch_id=5, row_idx=42)
//! );
//!
//! // Composite key
//! let key_cols = MoonlinkRow::new(vec![RowValue::Int32(100), RowValue::Int32(5)]);
//! index.insert(
//!     key=hash_of_key(&key_cols),
//!     identity_for_key=Some(key_cols),  // Need full identity
//!     RecordLocation::MemoryBatch(batch_id=3, row_idx=8)
//! );
//! ```
//!
//! ### Finding Records
//!
//! ```rust,ignore
//! let record = RawDeletionRecord {
//!     lookup_key: 12345,
//!     row_identity: None,
//!     ...
//! };
//! let locations = index.find_record(&record);  // Vec<RecordLocation>
//! ```
//!
//! ### Fast Deletion
//!
//! ```rust,ignore
//! // Remove and return location (for SinglePrimitive and Key only)
//! if let Some(location) = index.fast_delete(&record) {
//!     println!("Deleted record at {:?}", location);
//! }
//! ```
//!
//! ## Remapping During Flush
//!
//! When memory batches are flushed and compacted, row positions change.
//! The `remap_into_vec` method converts old locations to new ones:
//!
//! ```text
//! Before flush (3 memory batches):
//! Batch 0: [row0, row1, row2]  ← Index: key=100 at row=1
//! Batch 1: [row0, row1]         ← Index: key=200 at row=0
//! Batch 2: [row0, row1, row2]   ← Index: key=300 at row=2
//!
//! After compaction (1 parquet file, some rows filtered):
//! File 0: [batch0-row0, batch0-row2, batch2-row0, batch2-row1]
//!                        ↑
//!              Old: Batch 0, Row 1 → New: File 0, Row 1
//!
//! remap_into_vec produces:
//! [(100, file=0, row=1), (200, file=0, row=2), (300, file=0, row=3)]
//! ```

use crate::row::IdentityProp;
use crate::storage::index::*;

impl MemIndex {
    /// Create a new memory index based on table identity properties.
    ///
    /// Chooses the appropriate index variant:
    /// - Single primitive key → `SinglePrimitive` (HashTable, no duplicates)
    /// - Composite keys → `Key` (HashTable with identity, no duplicates)
    /// - Full row identity → `FullRow` (MultiMap, allows duplicates)
    /// - No identity → `None` (append-only, no index)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // For table: CREATE TABLE users (id INT PRIMARY KEY, name TEXT)
    /// let index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
    /// // Creates: MemIndex::SinglePrimitive(HashTable)
    ///
    /// // For table: CREATE TABLE items (order_id INT, item_id INT, PRIMARY KEY(order_id, item_id))
    /// let index = MemIndex::new(IdentityProp::Keys(vec![0, 1]));
    /// // Creates: MemIndex::Key(HashTable)
    ///
    /// // For table: CREATE TABLE events (...) -- no primary key, allows duplicates
    /// let index = MemIndex::new(IdentityProp::FullRow);
    /// // Creates: MemIndex::FullRow(MultiMap)
    ///
    /// // For table: Append-only time series
    /// let index = MemIndex::new(IdentityProp::None);
    /// // Creates: MemIndex::None
    /// ```
    pub fn new(identity: IdentityProp) -> Self {
        match identity {
            IdentityProp::SinglePrimitiveKey(_) => {
                MemIndex::SinglePrimitive(hashbrown::HashTable::new())
            }
            IdentityProp::Keys(_) => MemIndex::Key(hashbrown::HashTable::new()),
            IdentityProp::FullRow => MemIndex::FullRow(MultiMap::new()),
            IdentityProp::None => MemIndex::None,
        }
    }

    /// Create a new empty index of the same type as another index.
    ///
    /// Useful when creating a new memory batch that should have the same
    /// index structure as existing batches.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // First memory batch
    /// let batch1_index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
    ///
    /// // Create second batch with same index type
    /// let batch2_index = MemIndex::new_like(&batch1_index);
    /// // Both are SinglePrimitive, but batch2_index is empty
    ///
    /// assert!(batch2_index.is_empty());
    /// ```
    pub fn new_like(other: &MemIndex) -> Self {
        match other {
            MemIndex::SinglePrimitive(_) => MemIndex::SinglePrimitive(hashbrown::HashTable::new()),
            MemIndex::Key(_) => MemIndex::Key(hashbrown::HashTable::new()),
            MemIndex::FullRow(_) => MemIndex::FullRow(MultiMap::new()),
            MemIndex::None => MemIndex::None,
        }
    }

    /// Check if this index type allows duplicate keys.
    ///
    /// Returns:
    /// - `false` for SinglePrimitive and Key (enforces uniqueness)
    /// - `true` for FullRow (allows multiple rows with same hash)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let unique_index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
    /// assert!(!unique_index.allow_duplicate());
    ///
    /// let duplicate_index = MemIndex::new(IdentityProp::FullRow);
    /// assert!(duplicate_index.allow_duplicate());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if called on `MemIndex::None`.
    pub fn allow_duplicate(&self) -> bool {
        match self {
            MemIndex::SinglePrimitive(_) => false,
            MemIndex::Key(_) => false,
            MemIndex::FullRow(_) => true,
            MemIndex::None => panic!("AppendOnly index does not support duplicate checking"),
        }
    }

    /// Insert a record into the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The hash of the primary key (or the key itself for SinglePrimitive)
    /// * `identity_for_key` - Full key columns (required for Key variant, None otherwise)
    /// * `location` - Where the record is stored in the memory batch
    ///
    /// # Behavior by Variant
    ///
    /// ## SinglePrimitive
    /// - Uses `key` directly as both hash and storage key
    /// - `identity_for_key` must be None
    /// - Enforces uniqueness (duplicate insert will replace old value)
    ///
    /// ```rust,ignore
    /// let mut index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
    /// index.insert(
    ///     key=12345,  // user_id
    ///     identity_for_key=None,
    ///     RecordLocation::MemoryBatch(batch_id=5, row_idx=0)
    /// );
    /// ```
    ///
    /// ## Key (Composite)
    /// - Uses `key` as hash for lookup
    /// - `identity_for_key` must contain full key columns
    /// - Enforces uniqueness based on identity
    ///
    /// ```rust,ignore
    /// let mut index = MemIndex::new(IdentityProp::Keys(vec![0, 1]));
    /// let key_cols = MoonlinkRow::new(vec![
    ///     RowValue::Int32(100),  // order_id
    ///     RowValue::Int32(5),    // item_id
    /// ]);
    /// index.insert(
    ///     key=hash_function(&key_cols),
    ///     identity_for_key=Some(key_cols),
    ///     RecordLocation::MemoryBatch(batch_id=3, row_idx=42)
    /// );
    /// ```
    ///
    /// ## FullRow
    /// - Uses `key` as hash of entire row
    /// - `identity_for_key` must be None
    /// - Allows duplicates (same hash can map to multiple locations)
    ///
    /// ```rust,ignore
    /// let mut index = MemIndex::new(IdentityProp::FullRow);
    /// let row_hash = hash(entire_row_values);
    /// index.insert(
    ///     key=row_hash,
    ///     identity_for_key=None,
    ///     RecordLocation::MemoryBatch(batch_id=7, row_idx=0)
    /// );
    /// // Can insert again with same hash (duplicate event)
    /// index.insert(
    ///     key=row_hash,
    ///     identity_for_key=None,
    ///     RecordLocation::MemoryBatch(batch_id=7, row_idx=5)
    /// );
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if called on `MemIndex::None` (append-only tables).
    pub fn insert(
        &mut self,
        key: u64,
        identity_for_key: Option<MoonlinkRow>,
        location: RecordLocation,
    ) {
        match self {
            MemIndex::SinglePrimitive(map) => {
                assert!(identity_for_key.is_none());
                map.insert_unique(
                    key,
                    SinglePrimitiveKey {
                        hash: key,
                        location,
                    },
                    |k| k.hash,
                );
            }
            MemIndex::Key(map) => {
                let key_with_id = KeyWithIdentity {
                    hash: key,
                    identity: identity_for_key.unwrap(),
                    location,
                };
                map.insert_unique(key, key_with_id, |k| k.hash);
            }
            MemIndex::FullRow(map) => {
                assert!(identity_for_key.is_none());
                map.insert(key, location);
            }
            MemIndex::None => {
                panic!("AppendOnly index does not support insert operations")
            }
        }
    }

    /// Remove a record from the index and return its location.
    ///
    /// "Fast" because it removes in O(1) time without scanning. Only works for
    /// SinglePrimitive and Key variants (not FullRow which allows duplicates).
    ///
    /// # Arguments
    ///
    /// * `raw_record` - Record to delete
    ///
    /// # Returns
    ///
    /// - `Some(location)` if record was found and removed
    /// - `None` if record was not found
    ///
    /// # Example: SinglePrimitive
    ///
    /// ```rust,ignore
    /// let mut index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
    /// index.insert(100, None, RecordLocation::MemoryBatch(5, 42));
    ///
    /// let record = RawDeletionRecord { lookup_key: 100, ... };
    /// let location = index.fast_delete(&record);
    /// assert_eq!(location, Some(RecordLocation::MemoryBatch(5, 42)));
    ///
    /// // Subsequent delete returns None
    /// let location2 = index.fast_delete(&record);
    /// assert_eq!(location2, None);
    /// ```
    ///
    /// # Example: Composite Key (with Identity Verification)
    ///
    /// ```rust,ignore
    /// let mut index = MemIndex::new(IdentityProp::Keys(vec![0, 1]));
    /// let key = MoonlinkRow::new(vec![RowValue::Int32(100), RowValue::Int32(5)]);
    /// index.insert(hash(&key), Some(key.clone()), RecordLocation::MemoryBatch(3, 8));
    ///
    /// // Correct identity - deletes successfully
    /// let correct_record = RawDeletionRecord {
    ///     lookup_key: hash(&key),
    ///     row_identity: Some(key.clone()),
    ///     ...
    /// };
    /// assert!(index.fast_delete(&correct_record).is_some());
    ///
    /// // Wrong identity (even with same hash) - returns None
    /// let wrong_key = MoonlinkRow::new(vec![RowValue::Int32(200), RowValue::Int32(10)]);
    /// let wrong_record = RawDeletionRecord {
    ///     lookup_key: hash(&key),  // Same hash (collision)
    ///     row_identity: Some(wrong_key),  // Different identity
    ///     ...
    /// };
    /// assert!(index.fast_delete(&wrong_record).is_none());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if called on:
    /// - `MemIndex::FullRow` - use regular delete instead
    /// - `MemIndex::None` - append-only tables don't support deletion
    pub fn fast_delete(&mut self, raw_record: &RawDeletionRecord) -> Option<RecordLocation> {
        match self {
            MemIndex::SinglePrimitive(map) => {
                let entry = map.find_entry(raw_record.lookup_key, |key| {
                    key.hash == raw_record.lookup_key
                });
                if let Ok(entry) = entry {
                    Some(entry.remove().0.location)
                } else {
                    None
                }
            }
            MemIndex::Key(map) => {
                let entry = map.find_entry(raw_record.lookup_key, |k| {
                    k.hash == raw_record.lookup_key
                        && k.identity.values == raw_record.row_identity.as_ref().unwrap().values
                });
                if let Ok(entry) = entry {
                    Some(entry.remove().0.location)
                } else {
                    None
                }
            }
            MemIndex::FullRow(_) => {
                panic!("FullRow index does not support fast delete")
            }
            MemIndex::None => {
                panic!("AppendOnly index does not support delete operations")
            }
        }
    }

    /// Find all locations matching the given record.
    ///
    /// Searches this memory batch's index for records matching the lookup key
    /// and (if applicable) row identity.
    ///
    /// # Arguments
    ///
    /// * `raw_record` - Record to search for, containing:
    ///   - `lookup_key`: The hash of the primary key
    ///   - `row_identity`: Full key columns (for composite keys) or None
    ///
    /// # Returns
    ///
    /// - Empty vector if not found
    /// - Single-element vector for SinglePrimitive and Key variants
    /// - Multi-element vector for FullRow variant (allows duplicates)
    ///
    /// # Example: SinglePrimitive
    ///
    /// ```rust,ignore
    /// let mut index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
    /// index.insert(100, None, RecordLocation::MemoryBatch(5, 0));
    /// index.insert(200, None, RecordLocation::MemoryBatch(5, 1));
    ///
    /// let record = RawDeletionRecord {
    ///     lookup_key: 100,
    ///     row_identity: None,
    ///     ...
    /// };
    /// let locations = index.find_record(&record);
    /// assert_eq!(locations, vec![RecordLocation::MemoryBatch(5, 0)]);
    /// ```
    ///
    /// # Example: Composite Key
    ///
    /// ```rust,ignore
    /// let mut index = MemIndex::new(IdentityProp::Keys(vec![0, 1]));
    /// let key1 = MoonlinkRow::new(vec![RowValue::Int32(100), RowValue::Int32(5)]);
    /// index.insert(hash(&key1), Some(key1.clone()), RecordLocation::MemoryBatch(3, 0));
    ///
    /// // Must provide both hash and identity
    /// let record = RawDeletionRecord {
    ///     lookup_key: hash(&key1),
    ///     row_identity: Some(key1),
    ///     ...
    /// };
    /// let locations = index.find_record(&record);
    /// assert_eq!(locations.len(), 1);
    /// ```
    ///
    /// # Example: FullRow (Duplicates)
    ///
    /// ```rust,ignore
    /// let mut index = MemIndex::new(IdentityProp::FullRow);
    /// let row_hash = hash(entire_row);
    /// index.insert(row_hash, None, RecordLocation::MemoryBatch(7, 0));
    /// index.insert(row_hash, None, RecordLocation::MemoryBatch(7, 2));  // Duplicate!
    ///
    /// let record = RawDeletionRecord { lookup_key: row_hash, ... };
    /// let locations = index.find_record(&record);
    /// // Returns both locations
    /// assert_eq!(locations.len(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if called on `MemIndex::None` (append-only tables).
    pub fn find_record(&self, raw_record: &RawDeletionRecord) -> Vec<RecordLocation> {
        match self {
            MemIndex::SinglePrimitive(map) => {
                if let Some(entry) = map.find(raw_record.lookup_key, |key| {
                    key.hash == raw_record.lookup_key
                }) {
                    vec![entry.location.clone()]
                } else {
                    vec![]
                }
            }
            MemIndex::Key(map) => {
                if let Some(entry) = map.find(raw_record.lookup_key, |k| {
                    k.identity.values == raw_record.row_identity.as_ref().unwrap().values
                }) {
                    vec![entry.location.clone()]
                } else {
                    vec![]
                }
            }
            MemIndex::FullRow(map) => {
                if let Some(locations) = map.get_vec(&raw_record.lookup_key) {
                    locations.clone()
                } else {
                    vec![]
                }
            }
            MemIndex::None => panic!("AppendOnly index does not support record lookups"),
        }
    }

    /// Check if the index contains no entries.
    ///
    /// # Returns
    ///
    /// - `true` if no records are indexed
    /// - `true` always for `MemIndex::None` (append-only)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
    /// assert!(index.is_empty());
    ///
    /// index.insert(100, None, RecordLocation::MemoryBatch(5, 0));
    /// assert!(!index.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        match self {
            MemIndex::SinglePrimitive(map) => map.is_empty(),
            MemIndex::Key(map) => map.is_empty(),
            MemIndex::FullRow(map) => map.is_empty(),
            MemIndex::None => true, // Append-only tables are always considered "empty" for index purposes
        }
    }

    /// Remap index entries from old memory locations to new file locations.
    ///
    /// When memory batches are flushed and compacted into Parquet files, row positions
    /// change. This method translates old (batch_id, row_idx) references to new
    /// (file_segment_idx, row_idx) positions.
    ///
    /// # Arguments
    ///
    /// * `batch_id_to_idx` - Maps batch IDs to their index in the row_offset_mapping
    /// * `row_offset_mapping` - For each batch and row, the new location or None if filtered
    ///
    /// # Returns
    ///
    /// Vector of `(key, new_segment_idx, new_row_idx)` tuples for all records that
    /// survived compaction (not filtered out).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Setup: 3 memory batches being flushed
    /// let batch_id_to_idx = hashmap! {
    ///     42 => 0,  // Batch 42 is first in mapping
    ///     43 => 1,  // Batch 43 is second
    ///     44 => 2,  // Batch 44 is third
    /// };
    ///
    /// // Mapping shows new positions (or None if row was filtered)
    /// let row_offset_mapping = vec![
    ///     // Batch 42 (3 rows)
    ///     vec![
    ///         Some((0, 0)),  // Row 0 → Segment 0, Row 0
    ///         None,          // Row 1 filtered out (e.g., deleted)
    ///         Some((0, 1)),  // Row 2 → Segment 0, Row 1
    ///     ],
    ///     // Batch 43 (2 rows)
    ///     vec![
    ///         Some((0, 2)),  // Row 0 → Segment 0, Row 2
    ///         Some((1, 0)),  // Row 1 → Segment 1, Row 0
    ///     ],
    ///     // Batch 44 (1 row)
    ///     vec![
    ///         Some((1, 1)),  // Row 0 → Segment 1, Row 1
    ///     ],
    /// ];
    ///
    /// // Index contains:
    /// let mut index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
    /// index.insert(100, None, RecordLocation::MemoryBatch(42, 0));  // batch 42, row 0
    /// index.insert(200, None, RecordLocation::MemoryBatch(42, 1));  // batch 42, row 1 (will be filtered)
    /// index.insert(300, None, RecordLocation::MemoryBatch(43, 1));  // batch 43, row 1
    ///
    /// // Remap to new locations
    /// let remapped = index.remap_into_vec(&batch_id_to_idx, &row_offset_mapping);
    ///
    /// // Results (sorted for clarity):
    /// // [
    /// //   (100, 0, 0),  // key=100 now at segment 0, row 0
    /// //   (300, 1, 0),  // key=300 now at segment 1, row 0
    /// // ]
    /// // Note: key=200 is missing because row was filtered (mapped to None)
    /// ```
    ///
    /// # Use Case
    ///
    /// After flushing memory to disk:
    /// 1. Compact and filter rows (e.g., remove deleted records)
    /// 2. Write remaining rows to Parquet file(s)
    /// 3. Build new file index using remapped locations
    ///
    /// ```text
    /// Memory:
    /// Batch 42: [row0, row1, row2] with index entries
    ///   ↓ Flush & compact (row1 deleted)
    /// Disk:
    /// File 0: [row0, row2] with new index entries
    /// ```
    ///
    /// # Panics
    ///
    /// - If index contains `RecordLocation::DiskFile` (shouldn't happen in mem index)
    /// - If called on `MemIndex::None`
    pub fn remap_into_vec(
        &self,
        batch_id_to_idx: &std::collections::HashMap<u64, usize>,
        row_offset_mapping: &[Vec<Option<(usize, usize)>>],
    ) -> Vec<(u64, usize, usize)> {
        let remap = |key: u64, location: &RecordLocation| match location {
            RecordLocation::MemoryBatch(batch_id, row_idx) => {
                let old_location = (batch_id_to_idx[batch_id], row_idx);
                let new_location = row_offset_mapping[old_location.0][*old_location.1];
                new_location.map(|new_location| (key, new_location.0, new_location.1))
            }
            RecordLocation::DiskFile(_, _) => panic!("No disk file in mem index"),
        };

        match self {
            MemIndex::SinglePrimitive(map) => map
                .into_iter()
                .filter_map(|v| remap(v.hash, &v.location))
                .collect(),
            MemIndex::Key(map) => map
                .into_iter()
                .filter_map(|v| remap(v.hash, &v.location))
                .collect(),
            MemIndex::FullRow(map) => map.flat_iter().filter_map(|(k, v)| remap(*k, v)).collect(),
            MemIndex::None => panic!("AppendOnly index does not support remapping operations"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::row::RowValue;

    use super::*;

    #[test]
    fn test_fast_delete_with_single_primitive_key() {
        let mut mem_index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
        mem_index.insert(
            /*key=*/ 10,
            /*identity_for_key=*/ None,
            RecordLocation::MemoryBatch(0, 0),
        );

        // Delete for a non-existent entry.
        let record_loc = mem_index.fast_delete(&RawDeletionRecord {
            lookup_key: 0,
            row_identity: None,
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        });
        assert!(record_loc.is_none());

        // Delete for an existent entry.
        let deletion_record = RawDeletionRecord {
            lookup_key: 10,
            row_identity: None,
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        };
        let record_loc = mem_index.fast_delete(&deletion_record);
        assert!(matches!(
            record_loc.unwrap(),
            RecordLocation::MemoryBatch(_, _)
        ));

        // No entry left after a successful deletion.
        let record_loc = mem_index.fast_delete(&deletion_record);
        assert!(record_loc.is_none());
    }

    #[test]
    fn test_fast_delete_with_keys() {
        let existent_row = MoonlinkRow::new(vec![
            RowValue::Int32(1),
            RowValue::Float32(2.0),
            RowValue::ByteArray(b"abc".to_vec()),
        ]);

        let mut mem_index = MemIndex::new(IdentityProp::Keys(vec![0, 1]));
        mem_index.insert(
            /*key=*/ 10,
            /*identity_for_key=*/ Some(existent_row.clone()),
            RecordLocation::MemoryBatch(0, 0),
        );

        // Delete for a non-existent entry, with different lookup key.
        let non_existent_row = MoonlinkRow::new(vec![
            RowValue::Int32(2),
            RowValue::Float32(3.0),
            RowValue::ByteArray(b"bcd".to_vec()),
        ]);
        let record_loc = mem_index.fast_delete(&RawDeletionRecord {
            lookup_key: 0,
            row_identity: Some(non_existent_row.clone()),
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        });
        assert!(record_loc.is_none());

        // Delete for a non-existent entry, with the same key, but different row identity.
        let record_loc = mem_index.fast_delete(&RawDeletionRecord {
            lookup_key: 10,
            row_identity: Some(non_existent_row.clone()),
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        });
        assert!(record_loc.is_none());

        // Delete for an existent entry.
        let deletion_record = RawDeletionRecord {
            lookup_key: 10,
            row_identity: Some(existent_row.clone()),
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        };
        let record_loc = mem_index.fast_delete(&deletion_record);
        assert!(matches!(
            record_loc.unwrap(),
            RecordLocation::MemoryBatch(_, _)
        ));

        // No entry left after a successful deletion.
        let record_loc = mem_index.fast_delete(&deletion_record);
        assert!(record_loc.is_none());
    }

    #[tokio::test]
    async fn test_find_record_with_single_primitive_key() {
        let mut mem_index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
        mem_index.insert(
            /*key=*/ 10,
            /*identity_for_key=*/ None,
            RecordLocation::MemoryBatch(0, 0),
        );

        // Search for a non-existent entry.
        let record_locs = mem_index.find_record(&RawDeletionRecord {
            lookup_key: 0,
            row_identity: None,
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        });
        assert!(record_locs.is_empty());

        // Search for an existent entry.
        let deletion_record = RawDeletionRecord {
            lookup_key: 10,
            row_identity: None,
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        };
        let record_loc = mem_index.find_record(&deletion_record);
        assert_eq!(record_loc.len(), 1);
        assert!(matches!(record_loc[0], RecordLocation::MemoryBatch(_, _)));
    }

    #[tokio::test]
    async fn test_find_record_with_keys() {
        let existent_row = MoonlinkRow::new(vec![
            RowValue::Int32(1),
            RowValue::Float32(2.0),
            RowValue::ByteArray(b"abc".to_vec()),
        ]);

        let mut mem_index = MemIndex::new(IdentityProp::Keys(vec![0, 1]));
        mem_index.insert(
            /*key=*/ 10,
            /*identity_for_key=*/ Some(existent_row.clone()),
            RecordLocation::MemoryBatch(0, 0),
        );

        // Search for a non-existent entry, with different lookup key.
        let non_existent_row = MoonlinkRow::new(vec![
            RowValue::Int32(2),
            RowValue::Float32(3.0),
            RowValue::ByteArray(b"bcd".to_vec()),
        ]);
        let record_loc = mem_index.find_record(&RawDeletionRecord {
            lookup_key: 0,
            row_identity: Some(non_existent_row.clone()),
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        });
        assert!(record_loc.is_empty());

        // Search for a non-existent entry, with the same key, but different row identity.
        let record_loc = mem_index.find_record(&RawDeletionRecord {
            lookup_key: 10,
            row_identity: Some(non_existent_row.clone()),
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        });
        assert!(record_loc.is_empty());

        // Search for an existent entry.
        let deletion_record = RawDeletionRecord {
            lookup_key: 10,
            row_identity: Some(existent_row.clone()),
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        };
        let record_loc = mem_index.find_record(&deletion_record);
        assert_eq!(record_loc.len(), 1);
        assert!(matches!(record_loc[0], RecordLocation::MemoryBatch(_, _)));
    }

    #[tokio::test]
    async fn test_find_record_with_full_rows() {
        let existent_row = MoonlinkRow::new(vec![
            RowValue::Int32(1),
            RowValue::Float32(2.0),
            RowValue::ByteArray(b"abc".to_vec()),
        ]);

        let mut mem_index = MemIndex::new(IdentityProp::FullRow);
        mem_index.insert(
            /*key=*/ 10,
            /*identity_for_key=*/ None,
            RecordLocation::MemoryBatch(0, 0),
        );

        // Search for a non-existent entry, with different lookup key.
        let non_existent_row = MoonlinkRow::new(vec![
            RowValue::Int32(2),
            RowValue::Float32(3.0),
            RowValue::ByteArray(b"bcd".to_vec()),
        ]);
        let record_loc = mem_index.find_record(&RawDeletionRecord {
            lookup_key: 0,
            row_identity: Some(non_existent_row.clone()),
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        });
        assert!(record_loc.is_empty());

        // Search for an existent entry.
        let deletion_record = RawDeletionRecord {
            lookup_key: 10,
            row_identity: Some(existent_row.clone()),
            pos: None,
            lsn: 0,
            delete_if_exists: false,
        };
        let record_loc = mem_index.find_record(&deletion_record);
        assert_eq!(record_loc.len(), 1);
        assert!(matches!(record_loc[0], RecordLocation::MemoryBatch(_, _)));
    }

    #[test]
    fn test_append_only_mem_index() {
        let mem_index = MemIndex::new(IdentityProp::None);

        // Test that new_like creates another append-only index
        let new_index = MemIndex::new_like(&mem_index);
        assert!(matches!(new_index, MemIndex::None));

        // These operations should panic for AppendOnly index since they shouldn't be called
        // for append-only tables that don't use index-based operations
    }
}
