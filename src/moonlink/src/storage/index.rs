//! # Moonlink Index Module
//!
//! This module implements a two-tier indexing system for efficiently locating records
//! in Moonlink's hybrid storage architecture (memory + disk).
//!
//! ## Overview
//!
//! The indexing system enables fast lookups of records by their primary key or identity.
//! It consists of two main components:
//!
//! 1. **In-Memory Index**: Fast lookups for recently written data still in memory
//! 2. **File Index**: Persistent lookups for data flushed to disk (Parquet files)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │              MooncakeIndex                          │
//! │  (Main index coordinating all lookups)              │
//! └─────────────────┬───────────────────┬───────────────┘
//!                   │                   │
//!                   ▼                   ▼
//!        ┌──────────────────┐  ┌─────────────────────┐
//!        │  In-Memory Index │  │    File Indices     │
//!        │    (MemIndex)    │  │  (GlobalIndex)      │
//!        └──────────────────┘  └─────────────────────┘
//!                │                       │
//!                ▼                       ▼
//!        ┌──────────────┐        ┌─────────────────┐
//!        │Memory Batches│        │  Parquet Files  │
//!        │  (Recent)    │        │   (Flushed)     │
//!        └──────────────┘        └─────────────────┘
//! ```
//!
//! ## Index Types
//!
//! The system supports different index types based on table identity requirements:
//!
//! ### 1. SinglePrimitive
//! For tables with a single primitive column as the primary key.
//!
//! **Example**: User table with integer user_id
//! ```text
//! Key: 12345 → Location: MemoryBatch(0, row=42)
//! ```
//!
//! ### 2. Key (Composite Keys)
//! For tables with multiple columns forming the primary key.
//!
//! **Example**: Order items table with (order_id, item_id)
//! ```text
//! Hash: hash(order_id=100, item_id=5) → Row Identity + Location
//! ```
//!
//! ### 3. FullRow
//! For tables where the entire row serves as the identity (allows duplicates).
//!
//! **Example**: Event log where identical events can occur
//! ```text
//! Hash: hash(entire_row) → [Location1, Location2, ...]
//! ```
//!
//! ### 4. None (Append-Only)
//! For append-only tables where no lookups or deletions are needed.
//!
//! ## How Lookups Work
//!
//! When searching for a record:
//!
//! 1. **Hash the key**: Convert the lookup key to a 64-bit hash
//! 2. **Search in-memory indices**: Check all active memory batches
//! 3. **Search file indices**: Query persisted index blocks on disk
//! 4. **Return all matches**: Collect locations from both tiers
//!
//! **Example Lookup Flow**:
//! ```text
//! User requests: Find user_id = 12345
//!   ↓
//! Hash: splitmix64(12345) = 0x1a2b3c4d5e6f7890
//!   ↓
//! Check In-Memory: Found in MemoryBatch(2, row=15)
//!   ↓
//! Check File Indices: Found in DiskFile(file=3, row=892)
//!   ↓
//! Return: [MemoryBatch(2, 15), DiskFile(3, 892)]
//! ```
//!
//! ## Persistence and Caching
//!
//! File indices use memory-mapped I/O and integrate with the object storage cache
//! for fast access. See [`persisted_bucket_hash_map`] for details.
//!
//! ## Index Merging
//!
//! Small index files are periodically merged to avoid fragmentation.
//! See [`index_merge_config::FileIndexMergeConfig`] for configuration options.
//!
//! ## Modules
//!
//! - [`hash_index`]: Main index operations (insert, find, delete)
//! - [`mem_index`]: In-memory index implementation
//! - [`persisted_bucket_hash_map`]: Persistent file index with bucket hashing
//! - [`cache_utils`]: Integration with object storage cache
//! - [`index_merge_config`]: Configuration for index merge operations
//! - [`index_s3`]: S3-based index for cloud-native storage (no local disk)

pub mod cache_utils;
pub mod hash_index;
pub mod index_merge_config;
pub mod index_s3;
pub mod mem_index;
pub mod persisted_bucket_hash_map;

use crate::row::MoonlinkRow;
use crate::storage::storage_utils::{RawRecord, RecordLocation};
use multimap::MultiMap;
use persisted_bucket_hash_map::GlobalIndex;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The main index structure that coordinates lookups across both memory and disk.
///
/// `MooncakeIndex` maintains two separate index collections:
/// 1. In-memory indices for recent, unflushed data
/// 2. File indices for data that has been persisted to disk
///
/// # Structure
///
/// ```text
/// MooncakeIndex
/// ├─ in_memory_index: HashSet<Arc<MemIndex>>
/// │  └─ Each MemIndex covers one memory batch
/// └─ file_indices: Vec<GlobalIndex>
///    └─ Each GlobalIndex covers one or more Parquet files
/// ```
///
/// # Example Usage
///
/// ```rust,ignore
/// let mut index = MooncakeIndex::new();
///
/// // Insert an in-memory batch index
/// let mem_batch = Arc::new(MemIndex::new(IdentityProp::SinglePrimitiveKey(0)));
/// index.insert_memory_index(mem_batch.clone());
///
/// // Later, when data is flushed to disk, add file index
/// let file_index = builder.build_from_flush(hash_entries, file_id).await?;
/// index.insert_file_index(file_index);
///
/// // Now lookups will search both memory and disk
/// let locations = index.find_record(&deletion_record).await;
/// ```
///
/// # Lifecycle
///
/// 1. **Write**: New records go to memory batches with corresponding MemIndex
/// 2. **Flush**: Memory batches are written to Parquet files
/// 3. **Index Creation**: GlobalIndex is built for the new Parquet files
/// 4. **Merge**: Small file indices are periodically merged to reduce overhead
/// 5. **Lookup**: Queries search both in-memory and file indices
#[derive(Clone, Debug)]
pub struct MooncakeIndex {
    /// Collection of in-memory indices, one per active memory batch.
    ///
    /// Uses `IndexPtr` wrapper to enable pointer-based equality/hashing,
    /// allowing efficient removal of specific memory batch indices.
    pub(crate) in_memory_index: HashSet<IndexPtr>,

    /// Collection of file indices for persisted data.
    ///
    /// Each `FileIndex` (GlobalIndex) covers one or more Parquet files.
    /// Ordered from oldest to newest for efficient merging.
    pub(crate) file_indices: Vec<FileIndex>,
}

/// Type for primary keys used throughout the indexing system.
///
/// Primary keys are always represented as 64-bit unsigned integers,
/// either directly (for single primitive keys) or as hashes of composite keys.
pub type PrimaryKey = u64;

/// Entry for single primitive key indices.
///
/// Used when the table has a single column primary key with a primitive type (int, string, etc.).
///
/// # Example
///
/// For a users table with `user_id` as primary key:
/// ```text
/// user_id: 12345 → SinglePrimitiveKey {
///     hash: 12345,
///     location: MemoryBatch(batch_id=2, row_idx=42)
/// }
/// ```
#[derive(Clone, Debug)]
pub struct SinglePrimitiveKey {
    /// The primary key value (also serves as the hash)
    hash: PrimaryKey,
    /// Where the record is stored (memory batch or disk file)
    location: RecordLocation,
}

/// Entry for composite key indices.
///
/// Used when the table has multiple columns forming the primary key.
/// Stores both the hash (for lookup) and the full identity (for verification).
///
/// # Example
///
/// For an order_items table with (order_id, item_id) primary key:
/// ```text
/// Key: hash(order_id=100, item_id=5) → KeyWithIdentity {
///     hash: 0x1a2b3c4d,
///     identity: MoonlinkRow([100, 5]),
///     location: DiskFile(file_id=3, row_idx=892)
/// }
/// ```
#[derive(Clone, Debug)]
pub struct KeyWithIdentity {
    /// Hash of the composite key for fast lookup
    hash: PrimaryKey,
    /// The actual key columns needed to verify uniqueness
    identity: MoonlinkRow,
    /// Where the record is stored
    location: RecordLocation,
}

/// In-memory index for a single memory batch.
///
/// The variant chosen depends on the table's identity property:
/// - **SinglePrimitive**: One column, primitive type → No duplicates allowed
/// - **Key**: Multiple columns forming identity → No duplicates allowed  
/// - **FullRow**: Entire row is identity → Duplicates allowed (MultiMap)
/// - **None**: Append-only table → No index needed
///
/// # Example: Choosing the Right Variant
///
/// ```rust,ignore
/// // Single primitive key (e.g., user_id)
/// let index = MemIndex::new(IdentityProp::SinglePrimitiveKey(0));
/// // Results in: MemIndex::SinglePrimitive(HashTable)
///
/// // Composite key (e.g., order_id + item_id)
/// let index = MemIndex::new(IdentityProp::Keys(vec![0, 1]));
/// // Results in: MemIndex::Key(HashTable)
///
/// // Full row identity (e.g., event logs)
/// let index = MemIndex::new(IdentityProp::FullRow);
/// // Results in: MemIndex::FullRow(MultiMap) - allows duplicates
///
/// // Append-only (e.g., time-series data)
/// let index = MemIndex::new(IdentityProp::None);
/// // Results in: MemIndex::None - no lookups/deletes
/// ```
#[derive(Clone, Debug)]
pub enum MemIndex {
    /// Hash table for single primitive key (no duplicates)
    SinglePrimitive(hashbrown::HashTable<SinglePrimitiveKey>),

    /// Hash table for composite keys with identity verification (no duplicates)
    Key(hashbrown::HashTable<KeyWithIdentity>),

    /// Multi-map for full row identity (allows duplicates)
    FullRow(MultiMap<PrimaryKey, RecordLocation>),

    /// No index for append-only tables
    None,
}

/// Index for records persisted to Parquet files.
///
/// A `FileIndex` (alias for `GlobalIndex`) maps primary keys to their locations
/// in one or more Parquet data files using a persistent bucket-based hash map.
///
/// # Structure
///
/// ```text
/// FileIndex (GlobalIndex)
/// ├─ files: Vec<DataFile>           # Parquet files covered by this index
/// ├─ index_blocks: Vec<IndexBlock>  # Hash buckets stored in separate files
/// │  └─ Each block contains:
/// │     ├─ Bucket array: [offset₀, offset₁, ...]
/// │     └─ Entries: [(hash, file_id, row_idx), ...]
/// └─ Cached via memory-mapped I/O
/// ```
///
/// # Example
///
/// ```text
/// Given: Lookup key = 12345
/// Hash: splitmix64(12345) = 0x1a2b3c4d5e6f7890
///
/// Step 1: Determine bucket
///   bucket_idx = upper_bits(0x1a2b3c4d) % num_buckets
///   
/// Step 2: Find index block containing this bucket
///   block = index_blocks[2]  // bucket range [1000..2000]
///   
/// Step 3: Read bucket from mmap'd file
///   entry_offset = block.buckets[bucket_idx]
///   
/// Step 4: Scan entries at offset
///   For each entry:
///     if entry.hash == lower_bits(0x5e6f7890):
///       return RecordLocation::DiskFile(file_id=3, row_idx=892)
/// ```
///
/// See [`persisted_bucket_hash_map::GlobalIndex`] for implementation details.
pub type FileIndex = GlobalIndex;

/// Wrapper for `Arc<MemIndex>` that enables pointer-based equality and hashing.
///
/// This allows storing memory indices in a `HashSet` and removing specific ones
/// by pointer identity, even if their contents are identical.
///
/// # Why Pointer Identity?
///
/// When a memory batch is flushed to disk, we need to remove its specific `MemIndex`
/// from the active set. If we used value-based equality, we might accidentally
/// remove the wrong index if two batches happen to have identical contents.
///
/// # Example
///
/// ```rust,ignore
/// let index1 = Arc::new(MemIndex::new(IdentityProp::SinglePrimitiveKey(0)));
/// let index2 = Arc::new(MemIndex::new(IdentityProp::SinglePrimitiveKey(0)));
///
/// let ptr1 = IndexPtr(index1.clone());
/// let ptr2 = IndexPtr(index2.clone());
/// let ptr1_again = IndexPtr(index1.clone());
///
/// // Different Arc pointers, even with same content
/// assert_ne!(ptr1, ptr2);
///
/// // Same Arc pointer
/// assert_eq!(ptr1, ptr1_again);
/// ```
#[derive(Clone, Debug)]
pub(crate) struct IndexPtr(Arc<MemIndex>);

impl IndexPtr {
    pub fn arc_ptr(&self) -> Arc<MemIndex> {
        self.0.clone()
    }
}

impl PartialEq for IndexPtr {
    fn eq(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.0) == Arc::as_ptr(&other.0)
    }
}

impl Eq for IndexPtr {}

impl Hash for IndexPtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}
