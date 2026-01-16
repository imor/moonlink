//! # Persistent Bucket-Based Hash Map for File Indices
//!
//! This module implements a disk-based hash index for efficiently locating records
//! in Parquet data files. The index uses a bucket-based hash table with variable-bit
//! encoding for space efficiency.
//!
//! ## Terms
//!
//! ### Hash Entry
//!
//! A key value pair stored in the index. Also called an entry in short.
//! The key stored in the entry is the lower bits of the hash of the
//! original key (primary key). The value is the file id and row id in
//! the file.
//!
//! ### Hash Bucket
//!
//! A container for multiple entries. All entries in the bucket have the
//! same upper half bits of the hash of the key.
//!
//! ### Index Block
//!
//! An index block contains multiple hash buckets.
//!
//! ## Architecture Overview
//!
//! ```text
//! GlobalIndex (covers multiple Parquet files)
//! ├─ files: [file1.parquet, file2.parquet, file3.parquet]
//! └─ index_blocks: [block1.idx, block2.idx]
//!    ├─ Each block contains a portion of the hash buckets
//!    └─ Memory-mapped for fast random access
//! ```
//!
//! ## Data Structure
//!
//! Each index block file has two sections:
//!
//! ### 1. Bucket Array
//!
//! A fixed-size array mapping bucket indices to entry offsets:
//!
//! ```text
//! Bucket Array (variable-bit encoding):
//! [offset_0, offset_1, offset_2, , offset_N]
//!     │       │       │            │
//!     v       v       v            v
//!   Points to entries in the entries section
//! ```
//!
//! ### 2. Entries Section
//!
//! Hash entries stored consecutively:
//!
//! ```text
//! Entries (variable-bit encoding):
//! [(lower_hash, file_idx, row_idx), (lower_hash, file_idx, row_idx)]...
//!  │          │        │
//!  │          │        └─ Which row in the file
//!  │          └─ Which file (index into files array)
//!  └─ Lower bits of hash (upper bits = bucket index)
//! ```
//!
//! ## Hash Splitting
//!
//! Each 64-bit hash is split into upper and lower parts:
//!
//! ```text
//! Hash: 0x1A2B3C4D_5E6F7890
//!       ├────────┘└────────┤
//!       Upper bits   Lower bits
//!       (bucket)     (stored in entry)
//! ```
//!
//! **Why?** Upper bits select the bucket; lower bits verify the match.
//! This saves space by not storing the full 64-bit hash in each entry.
//!
//! ## Lookup Process
//!
//! ### Example: Find key = 12345
//!
//! **Step 1: Hash the key**
//! ```text
//! hash = splitmix64(12345) = 0x1A2B3C4D5E6F7890
//! ```
//!
//! **Step 2: Split the hash**
//! ```text
//! upper_bits = 0x1A2B3C4D  (bucket index)
//! lower_bits = 0x5E6F7890  (for verification)
//! ```
//!
//! **Step 3: Find the bucket**
//! ```text
//! bucket_idx = upper_bits % num_buckets = 0x1A2B3C4D % 1000 = 623
//! ```
//!
//! **Step 4: Determine which index block contains this bucket**
//! ```text
//! index_blocks:
//!   block_0: buckets [0..500)
//!   block_1: buckets [500..1000)  ← Contains bucket 623
//! ```
//!
//! **Step 5: Read bucket entry**
//! ```text
//! mmap file: block_1.idx
//! Seek to: bucket_623_offset
//! Read: entry_start=1000, entry_end=1003
//! ```
//!
//! **Step 6: Scan entries in bucket**
//! ```text
//! Entries [1000..1003):
//!   entry_1000: hash=0x5E6F7000, file=2, row=100
//!   entry_1001: hash=0x5E6F7890, file=3, row=892  ← Match!
//!   entry_1002: hash=0x5E6F7FFF, file=1, row=50
//! ```
//!
//! **Step 7: Return result**
//! ```text
//! Found: RecordLocation::DiskFile(file_id=3, row_idx=892)
//! ```
//!
//! ## Variable-Bit Encoding
//!
//! The index uses variable-bit encoding to minimize size:
//!
//! ```rust,ignore
//! // Example: If we have 4 files and 1 million rows per file
//! seg_id_bits = ceil(log2(4)) = 2 bits
//! row_id_bits = ceil(log2(1_000_000)) = 20 bits
//! hash_lower_bits = 64 - hash_upper_bits
//!
//! // Each entry uses only: hash_lower_bits + 2 + 20 bits
//! // Instead of: 64 + 32 + 32 = 128 bits
//! ```
//!
//! ## Index Blocks
//!
//! Large indices are split into multiple blocks for:
//! - **Parallel access**: Different queries can read different blocks
//! - **Cache efficiency**: Only load needed blocks into cache
//! - **Incremental building**: Add new blocks without rewriting everything
//!
//! ## Building Indices
//!
//! ### From Flush (New Data)
//!
//! ```rust,ignore
//! let file_index = GlobalIndexBuilder::new()
//!     .set_files(vec![parquet_file1, parquet_file2])
//!     .set_directory(index_dir)
//!     .build_from_flush(
//!         vec![(key1, file_idx, row_idx), (key2, file_idx, row_idx)],
//!         file_id
//!     )
//!     .await?;
//! ```
//!
//! ### From Merge (Combining Existing Indices)
//!
//! ```rust,ignore
//! let merged_index = GlobalIndexBuilder::new()
//!     .set_directory(index_dir)
//!     .build_from_merge(&[index1, index2, index3], file_id)
//!     .await?;
//! ```
//!
//! ## Hash Function
//!
//! Uses `splitmix64` - a fast, high-quality hash function:
//! - Good avalanche properties (small input change = large hash change)
//! - Fast computation (few operations)
//! - Good distribution (avoids clustering in buckets)

use crate::create_data_file;
use crate::error::Result;
use crate::storage::async_bitwriter::BitWriter as AsyncBitWriter;
use crate::storage::storage_utils::{MooncakeDataFileRef, RecordLocation};
use crate::NonEvictableHandle;
use bitstream_io::{BigEndian, BitRead, BitReader};
use memmap2::Mmap;
use std::collections::{BinaryHeap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::io::SeekFrom;
use std::path::PathBuf;
use std::sync::Arc;
use std::{fmt, vec};
use tokio::fs::File as AsyncFile;
use tokio_bitstream_io::BigEndian as AsyncBigEndian;

// Constants
const HASH_BITS: u32 = 64;
// const _MAX_BLOCK_SIZE: u32 = 2 * 1024 * 1024 * 1024; // 2GB
// const _TARGET_NUM_FILES_PER_INDEX: u32 = 4000;
const INVALID_FILE_ID: u32 = 0xFFFFFFFF;

/// High-quality hash function for distributing keys across buckets.
///
/// SplitMix64 is a fast, non-cryptographic hash function with excellent
/// statistical properties. It's used to hash primary keys before indexing.
///
/// # Properties
///
/// - **Fast**: Only a few arithmetic operations
/// - **Good distribution**: Avoids clustering in hash buckets
/// - **Avalanche**: Small input changes cause large output changes
///
/// # Example
///
/// ```rust,ignore
/// let key = 12345u64;
/// let hash = splitmix64(key);
/// // hash = 0x1a2b3c4d5e6f7890 (64-bit)
///
/// // Even nearby keys produce very different hashes
/// let hash2 = splitmix64(12346);
/// // hash2 = 0x9f8e7d6c5b4a3920 (completely different)
/// ```
pub(super) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Persistent hash index for locating records in Parquet files.
///
/// A `GlobalIndex` provides O(1) average-case lookup of records across one or more
/// Parquet data files using a bucket-based hash table stored on disk.
///
/// # Structure
///
/// ```text
/// GlobalIndex
/// ├─ files: Vec<DataFile>          # Parquet files indexed
/// ├─ num_rows: u32                  # Total rows across all files
/// ├─ hash_bits: u32                 # Total hash bits (always 64)
/// ├─ hash_upper_bits: u32           # Bits for bucket index
/// ├─ hash_lower_bits: u32           # Bits stored in entries
/// ├─ seg_id_bits: u32               # Bits for file index
/// ├─ row_id_bits: u32               # Bits for row index
/// ├─ bucket_bits: u32               # Bits for bucket offsets
/// └─ index_blocks: Vec<IndexBlock>  # Physical index files
/// ```
///
/// # Example: Small Index
///
/// ```rust,ignore
/// // Indexing 2 files with 1000 rows each
/// GlobalIndex {
///     files: [file1.parquet, file2.parquet],
///     num_rows: 2000,
///     
///     // Bit allocations
///     hash_bits: 64,
///     hash_upper_bits: 32,  // 4 billion buckets possible
///     hash_lower_bits: 32,  // Verify with lower 32 bits
///     seg_id_bits: 1,       // 2 files → 1 bit
///     row_id_bits: 10,      // 1024 rows max → 10 bits
///     bucket_bits: 11,      // Entry counts up to 2048 → 11 bits
///     
///     index_blocks: [block0.idx],  // Single block for small index
/// }
/// ```
///
/// # Bit Allocation Strategy
///
/// The index calculates optimal bit widths based on data size:
///
/// ```rust,ignore
/// // For 16 files with up to 10 million rows each
/// seg_id_bits = ceil(log2(16)) = 4 bits
/// row_id_bits = ceil(log2(10_000_000)) = 24 bits
///
/// // Each entry: hash_lower + seg_id + row_id
/// //           = 32 + 4 + 24 = 60 bits (vs 128 bits naive encoding)
/// ```
///
/// # Hash Index
///
/// Hash index that maps a u64 key to [file_idx, row_idx].
///
/// The physical structure on disk:
///
/// ```text
/// ┌────────────────────────────────────────┐
/// │           Bucket Array                   │
/// │ [offset_0][offset_1]...[offset_N]      │
/// └────────────────────────────────────────┘
/// ┌────────────────────────────────────────┐
/// │              Entries                     │
/// │ [lower_hash, file_idx, row_idx]...     │
/// └────────────────────────────────────────┘
/// ```
#[derive(Clone)]
pub struct GlobalIndex {
    pub(crate) files: Vec<MooncakeDataFileRef>,
    pub(crate) num_rows: u32,
    pub(crate) hash_bits: u32,
    pub(crate) hash_upper_bits: u32,
    pub(crate) hash_lower_bits: u32,
    pub(crate) seg_id_bits: u32,
    pub(crate) row_id_bits: u32,
    pub(crate) bucket_bits: u32,

    pub(crate) index_blocks: Vec<IndexBlock>,
}

// For GlobalIndex, there won't be two indices pointing to same sets of data files, so we use data files for hash and equal.
impl PartialEq for GlobalIndex {
    fn eq(&self, other: &Self) -> bool {
        self.files == other.files
    }
}

impl Eq for GlobalIndex {}

/// It's guaranteed every file index references to different set of data files.
impl Hash for GlobalIndex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.files.hash(state);
    }
}

/// A single index block file containing a subset of hash buckets.
///
/// Large indices are split across multiple blocks for:
/// - **Parallel access**: Different threads can read different blocks
/// - **Cache efficiency**: Load only needed blocks into memory
/// - **Incremental updates**: Add new blocks without rewriting all data
///
/// # Structure
///
/// ```text
/// IndexBlock
/// ├─ bucket_start_idx: 0           # First bucket in this block
/// ├─ bucket_end_idx: 1000          # Last bucket (exclusive)
/// ├─ bucket_start_offset: 0        # Bit offset to bucket array
/// ├─ index_file: "block_0.idx"     # Physical file
/// ├─ file_size: 5242880            # File size (for merge decisions)
/// ├─ data: Some(Mmap)              # Memory-mapped file content
/// └─ cache_handle: Some(Handle)    # Keeps block in cache
/// ```
///
/// # Example
///
/// ```text
/// GlobalIndex with 3 index blocks covering 3000 buckets:
///
/// block_0: buckets [0..1000), file="idx_0.bin", size=4MB
/// block_1: buckets [1000..2000), file="idx_1.bin", size=4MB  
/// block_2: buckets [2000..3000), file="idx_2.bin", size=3MB
///
/// Lookup for bucket 1500:
///   → Use block_1 (contains buckets 1000..2000)
///   → Mmap idx_1.bin
///   → Read bucket at index 500 (relative to block start)
/// ```
#[derive(Clone)]
pub(crate) struct IndexBlock {
    /// Index of the first bucket (inclusive) in this block
    pub(crate) bucket_start_idx: u32,

    /// Index of the last bucket (exclusive) in this block
    pub(crate) bucket_end_idx: u32,

    /// Byte offset where the buckets array starts
    pub(crate) bucket_start_offset: u64,

    /// Local index file path.
    pub(crate) index_file: MooncakeDataFileRef,

    /// File size for the index block file, used to decide whether to trigger merge index blocks merge.
    pub(crate) file_size: u64,

    /// Mmapped-data.
    /// Synchronous IO is not needed because here we use mmap.
    data: Arc<Option<Mmap>>,

    /// Cache handle within object storage cache.
    pub(crate) cache_handle: Option<NonEvictableHandle>,
}

/// Represents a bucket's hash range and entry positions.
///
/// Used during lookup to identify which entries need to be scanned.
///
/// # Fields
///
/// * `upper_hash` - Upper bits of hash (bucket identifier) shifted to full hash position
/// * `entry_start` - Index of first entry in this bucket
/// * `entry_end` - Index after last entry (exclusive)
struct BucketEntry {
    upper_hash: u64,
    entry_start: u32,
    entry_end: u32,
}

impl IndexBlock {
    /// Create an index block by memory-mapping an existing index file.
    ///
    /// # Arguments
    ///
    /// * `bucket_start_idx` - First bucket covered by this block
    /// * `bucket_end_idx` - Last bucket (exclusive) covered by this block
    /// * `bucket_start_offset` - Bit offset where bucket array starts in file
    /// * `index_file` - Reference to the index block file
    pub(crate) async fn new(
        bucket_start_idx: u32,
        bucket_end_idx: u32,
        bucket_start_offset: u64,
        index_file: MooncakeDataFileRef,
    ) -> Self {
        let file = tokio::fs::File::open(index_file.file_path()).await.unwrap();
        let file_metadata = file.metadata().await.unwrap();
        let file = file.into_std().await;
        let data = unsafe { Mmap::map(&file).unwrap() };
        Self {
            bucket_start_idx,
            bucket_end_idx,
            bucket_start_offset,
            index_file,
            file_size: file_metadata.len(),
            data: Arc::new(Some(data)),
            cache_handle: None,
        }
    }

    /// Create an iterator over all entries in this index block.
    ///
    /// # Arguments
    ///
    /// * `global_index` - Global index metadata for bit widths
    /// * `file_id_remap` - Maps old file IDs to new file IDs
    fn create_iterator<'a>(
        &'a self,
        global_index: &'a GlobalIndex,
        file_id_remap: &'a Vec<u32>,
    ) -> IndexBlockIterator<'a> {
        IndexBlockIterator::new(self, global_index, file_id_remap)
    }

    /// Read bucket metadata for specified bucket indices.
    ///
    /// Extracts the entry range for each requested bucket by reading
    /// the bucket array.
    ///
    /// # Arguments
    ///
    /// * `bucket_idxs` - Bucket indices to read
    /// * `reader` - Bit reader positioned at bucket array
    /// * `global_index` - Global index metadata for bucket bit width
    ///
    /// # Returns
    ///
    /// Vector of `BucketEntry` structs for non-empty buckets.
    #[inline]
    fn read_buckets(
        &self,
        bucket_idxs: &[u32],
        reader: &mut BitReader<Cursor<&[u8]>, BigEndian>,
        global_index: &GlobalIndex,
    ) -> Vec<BucketEntry> {
        let mut results = Vec::new();
        for bucket_idx in bucket_idxs {
            reader
                .seek_bits(SeekFrom::Start(
                    self.bucket_start_offset + (bucket_idx * global_index.bucket_bits) as u64,
                ))
                .unwrap();
            let start = reader
                .read_unsigned_var::<u32>(global_index.bucket_bits)
                .unwrap();
            let end = reader
                .read_unsigned_var::<u32>(global_index.bucket_bits)
                .unwrap();
            if start != end {
                results.push(BucketEntry {
                    upper_hash: (*bucket_idx as u64) << global_index.hash_lower_bits,
                    entry_start: start,
                    entry_end: end,
                });
            }
        }
        results
    }

    /// Read a single hash entry from the index.
    ///
    /// Decodes variable-width fields: lower hash bits, segment index, and row index.
    ///
    /// # Arguments
    ///
    /// * `reader` - Bit reader positioned at entry
    /// * `global_index` - Global index metadata for field bit widths
    ///
    /// # Returns
    ///
    /// Tuple of (lower_hash, seg_idx, row_idx)
    #[inline]
    fn read_entry(
        &self,
        reader: &mut BitReader<Cursor<&[u8]>, BigEndian>,
        global_index: &GlobalIndex,
    ) -> (u64, usize, usize) {
        let hash = reader
            .read_unsigned_var::<u64>(global_index.hash_lower_bits)
            .unwrap();
        let seg_idx = reader
            .read_unsigned_var::<u32>(global_index.seg_id_bits)
            .unwrap();
        let row_idx = reader
            .read_unsigned_var::<u32>(global_index.row_id_bits)
            .unwrap();
        (hash, seg_idx as usize, row_idx as usize)
    }

    /// Search for multiple values in this index block.
    ///
    /// Reads specified buckets and scans their entries, matching against
    /// the provided hashes.
    ///
    /// # Arguments
    ///
    /// * `value_and_hashes` - Pairs of (original_key, hash) sorted by hash
    /// * `bucket_idxs` - Bucket indices to scan
    /// * `global_index` - Global index metadata
    ///
    /// # Returns
    ///
    /// Vector of (original_key, location) for all matches found.
    fn read(
        &self,
        value_and_hashes: &[(u64, u64)],
        mut bucket_idxs: Vec<u32>,
        global_index: &GlobalIndex,
    ) -> Vec<(u64, RecordLocation)> {
        let cursor = Cursor::new(self.data.as_ref().as_ref().unwrap().as_ref());
        let mut reader = BitReader::endian(cursor, BigEndian);
        let mut entry_reader = reader.clone();
        bucket_idxs.dedup();
        let bucket_entries = self.read_buckets(&bucket_idxs, &mut reader, global_index);

        let mut results = Vec::new();
        let mut lookup_iter =
            LookupIterator::new(self, global_index, &mut entry_reader, &bucket_entries);
        let mut i = 0;
        let mut lookup_entry = lookup_iter.next();
        while let Some((entry_hash, seg_idx, row_idx)) = lookup_entry {
            while i < value_and_hashes.len() && value_and_hashes[i].1 < entry_hash {
                i += 1;
            }
            if i < value_and_hashes.len() && value_and_hashes[i].1 == entry_hash {
                let value = value_and_hashes[i].0;
                results.push((
                    value,
                    RecordLocation::DiskFile(global_index.files[seg_idx].file_id(), row_idx),
                ));
            }
            lookup_entry = lookup_iter.next();
        }
        results
    }
}

/// Iterator for scanning hash entries within specific buckets during lookup.
///
/// This iterator efficiently walks through hash entries stored in specific
/// buckets without loading the entire index into memory.
///
/// # Structure
///
/// ```text
/// LookupIterator
/// ├─ entries: [Bucket623, Bucket892, Bucket1005]  # Buckets to scan
/// ├─ current_bucket: 0                             # Currently scanning Bucket623
/// ├─ current_entry: 2                              # On entry 2 of current bucket
/// └─ entry_reader: BitReader                       # Reads hash entries from mmap
/// ```
///
/// # Usage Pattern
///
/// ```text
/// Step 1: Create iterator with specific buckets to scan
/// Step 2: Iterator seeks to first entry of first bucket
/// Step 3: next() returns (full_hash, seg_idx, row_idx)
/// Step 4: When bucket exhausted, moves to next bucket
/// Step 5: Continues until all specified buckets scanned
/// ```
///
/// # Example
///
/// ```rust,ignore
/// // Looking up keys that hash to buckets [100, 100, 205]
/// let buckets = vec![
///     BucketEntry { upper_hash: 100 << 32, entry_start: 50, entry_end: 53 },
///     BucketEntry { upper_hash: 205 << 32, entry_start: 892, entry_end: 895 },
/// ];
///
/// let mut iter = LookupIterator::new(index_block, metadata, &mut reader, &buckets);
///
/// // Scan bucket 100 (entries 50, 51, 52)
/// while let Some((hash, seg, row)) = iter.next() {
///     // Compare hash against lookup keys
/// }
/// // Then automatically scans bucket 205 (entries 892, 893, 894)
/// ```
pub struct LookupIterator<'a> {
    index: &'a IndexBlock,
    metadata: &'a GlobalIndex,
    entry_reader: &'a mut BitReader<Cursor<&'a [u8]>, BigEndian>,
    entries: &'a Vec<BucketEntry>,
    current_bucket: usize,
    current_entry: u32,
}

impl<'a> LookupIterator<'a> {
    /// Create a new lookup iterator for specific buckets.
    ///
    /// Initializes the iterator and seeks to the first entry of the first bucket.
    ///
    /// # Arguments
    ///
    /// * `index` - Index block containing the entries
    /// * `metadata` - Global index metadata for bit widths
    /// * `entry_reader` - Bit reader positioned to read entries
    /// * `entries` - Bucket entries to scan (bucket ranges)
    fn new(
        index: &'a IndexBlock,
        metadata: &'a GlobalIndex,
        entry_reader: &'a mut BitReader<Cursor<&'a [u8]>, BigEndian>,
        entries: &'a Vec<BucketEntry>,
    ) -> Self {
        let mut ret = Self {
            index,
            metadata,
            entry_reader,
            entries,
            current_bucket: 0,
            current_entry: 0,
        };
        ret.seek_to_bucket_entry_start();
        ret
    }

    /// Seek the entry reader to the start of the current bucket's entries.
    ///
    /// Calculates the bit offset of the first entry in the current bucket
    /// and positions the reader there.
    ///
    /// # Calculation
    ///
    /// ```text
    /// entry_size = hash_lower_bits + seg_id_bits + row_id_bits
    /// bit_offset = current_entry * entry_size
    /// ```
    fn seek_to_bucket_entry_start(&mut self) {
        if self.current_bucket < self.entries.len() {
            self.current_entry = self.entries[self.current_bucket].entry_start;
            self.entry_reader
                .seek_bits(SeekFrom::Start(
                    self.current_entry as u64
                        * (self.metadata.hash_lower_bits
                            + self.metadata.seg_id_bits
                            + self.metadata.row_id_bits) as u64,
                ))
                .unwrap();
        }
    }

    /// Get the next hash entry from the current bucket.
    ///
    /// Reads entries sequentially within the current bucket. When a bucket
    /// is exhausted, automatically advances to the next bucket.
    ///
    /// # Returns
    ///
    /// * `Some((full_hash, seg_idx, row_idx))` - Next entry
    /// * `None` - All buckets exhausted
    ///
    /// # Hash Reconstruction
    ///
    /// ```text
    /// Stored: lower_hash = 0x5E6F7890 (32 bits)
    /// Bucket: upper_hash = 0x1A2B3C4D00000000 (from bucket index)
    /// Result: full_hash  = 0x1A2B3C4D5E6F7890 (64 bits)
    /// ```
    fn next(&mut self) -> Option<(u64, usize, usize)> {
        loop {
            if self.current_bucket >= self.entries.len() {
                return None;
            }
            if self.current_entry < self.entries[self.current_bucket].entry_end {
                let (lower_hash, seg_idx, row_idx) =
                    self.index.read_entry(self.entry_reader, self.metadata);
                self.current_entry += 1;
                return Some((
                    lower_hash | self.entries[self.current_bucket].upper_hash,
                    seg_idx,
                    row_idx,
                ));
            }
            self.current_bucket += 1;
            self.seek_to_bucket_entry_start();
        }
    }
}

impl GlobalIndex {
    /// Get the total size of all index block files.
    ///
    /// Used to determine if index merging should be triggered.
    ///
    /// # Returns
    ///
    /// Total bytes across all index block files.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let total_size = index.get_index_blocks_size();
    /// if total_size < merge_config.index_block_final_size {
    ///     // This index is small enough to be merged
    ///     candidate_indices.push(index);
    /// }
    /// ```
    pub fn get_index_blocks_size(&self) -> u64 {
        self.index_blocks
            .iter()
            .map(|cur_index_block| cur_index_block.file_size)
            .sum()
    }
    /// Search for multiple values efficiently in a single pass.
    ///
    /// More efficient than repeated single lookups because it:
    /// 1. Processes all values in one scan of index blocks
    /// 2. Amortizes the cost of reading and decompressing index data
    ///
    /// # Arguments
    ///
    /// * `value_and_hashes` - Pairs of (original_key, hash(key)) sorted by hash
    ///
    /// # Returns
    ///
    /// Vector of (original_key, location) for all matches found.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// Input: [(key1, hash1), (key2, hash2), (key3, hash3)]
    ///         Sorted by hash
    ///
    /// For each index block:
    ///   1. Determine which hashes fall in this block's bucket range
    ///   2. Extract upper bits to get bucket indices
    ///   3. Read those buckets from mmap'd file
    ///   4. Scan entries, matching against lower hash bits
    ///   5. Collect all matches
    ///
    /// Combine results from all blocks
    /// ```
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Prepare hashes for multiple keys
    /// let keys = vec![100u64, 200, 300];
    /// let value_and_hashes: Vec<_> = keys.iter()
    ///     .map(|k| (*k, splitmix64(*k)))
    ///     .collect();
    ///
    /// // Must be sorted by hash
    /// value_and_hashes.sort_by_key(|(_, hash)| *hash);
    ///
    /// // Batch lookup
    /// let results = index.search_values(&value_and_hashes).await;
    ///
    /// // Results:
    /// // [
    /// //   (100, DiskFile(file=2, row=50)),
    /// //   (300, DiskFile(file=1, row=892)),
    /// // ]
    /// // Note: key=200 not in results (not found)
    /// ```
    ///
    /// # Performance
    ///
    /// **Single lookups** (search_values called 3 times with 1 key each):
    /// ```text
    /// Each call: Scan all index blocks, read buckets, scan entries
    /// Total: 3 * O(num_blocks)
    /// ```
    ///
    /// **Batch lookup** (search_values called once with 3 keys):
    /// ```text
    /// One pass: Scan all index blocks once, read relevant buckets, scan entries
    /// Total: O(num_blocks)
    /// ```
    ///
    /// Speedup: ~3x for this example, scales with batch size.
    pub async fn search_values(
        &self,
        value_and_hashes: &[(u64, u64)],
    ) -> Vec<(u64, RecordLocation)> {
        let mut results = Vec::new();
        let upper_hashes = value_and_hashes
            .iter()
            .map(|(_, hash)| (hash >> self.hash_lower_bits) as u32)
            .collect::<Vec<_>>();
        let mut start_idx = 0;
        for block in self.index_blocks.iter() {
            while upper_hashes[start_idx] < block.bucket_start_idx {
                start_idx += 1;
            }
            let mut end_idx = start_idx;
            while end_idx < upper_hashes.len() && upper_hashes[end_idx] < block.bucket_end_idx {
                end_idx += 1;
            }
            results.extend(block.read(
                &value_and_hashes[start_idx..end_idx],
                upper_hashes[start_idx..end_idx].to_vec(),
                self,
            ));
        }
        results
    }

    /// Create an iterator over all hash entries in the index.
    ///
    /// # Arguments
    ///
    /// * `file_id_remap` - Maps old file indices to new file indices (for merging)
    ///
    /// # Returns
    ///
    /// Iterator that yields (hash, seg_idx, row_idx) tuples in hash order.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // No remapping needed
    /// let identity_remap = (0..index.files.len() as u32).collect();
    /// let mut iter = index.create_iterator(&identity_remap);
    ///
    /// while let Some((hash, seg_idx, row_idx)) = iter.next() {
    ///     println!("Entry: hash={}, file={}, row={}", hash, seg_idx, row_idx);
    /// }
    /// ```
    pub fn create_iterator<'a>(&'a self, file_id_remap: &'a Vec<u32>) -> GlobalIndexIterator<'a> {
        GlobalIndexIterator::new(self, file_id_remap)
    }

    /// Prepare a batch of keys for lookup by hashing and sorting.
    ///
    /// Converts keys to (key, hash) pairs and sorts by hash for efficient
    /// batch lookup via `search_values`.
    ///
    /// # Arguments
    ///
    /// * `values` - Iterator of u64 keys to look up
    ///
    /// # Returns
    ///
    /// Vector of (key, hash) pairs sorted by hash.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let keys = vec![12345u64, 67890, 11111, 99999];
    /// let value_and_hashes = GlobalIndex::prepare_hashes_for_lookup(keys.iter().copied());
    ///
    /// // Result (sorted by hash, not by key):
    /// // [
    /// //   (11111, splitmix64(11111)),
    /// //   (67890, splitmix64(67890)),
    /// //   (12345, splitmix64(12345)),
    /// //   (99999, splitmix64(99999)),
    /// // ]
    ///
    /// // Now ready for batch lookup
    /// let results = index.search_values(&value_and_hashes).await;
    /// ```
    ///
    /// # Why Sort?
    ///
    /// Sorting enables efficient sequential scanning of index blocks:
    /// ```text
    /// Unsorted hashes: Need to scan entire index for each hash
    /// Sorted hashes: Scan index once, matching hashes in order
    /// ```
    pub fn hash_sort_and_dedup(values: impl Iterator<Item = u64>) -> Vec<(u64, u64)> {
        let mut ret = values
            .map(|value| (value, splitmix64(value)))
            .collect::<Vec<_>>();
        ret.sort_unstable_by_key(|(_, hash)| *hash);
        ret.dedup_by_key(|(_, hash)| *hash);
        ret
    }
}

// ================================
// Builders
// ================================

/// Builder for creating a single index block file.
///
/// Writes hash entries to disk in two phases:
/// 1. **Entry writing**: Append hash entries as they arrive
/// 2. **Bucket writing**: Write bucket array pointing to entries
///
/// # Structure
///
/// ```text
/// IndexBlockBuilder
/// ├─ bucket_start_idx: 0           # First bucket this block covers
/// ├─ bucket_end_idx: 1000          # Last bucket (exclusive)
/// ├─ buckets: [0, 0, 3, 3, 7, ...] # Entry counts per bucket
/// ├─ current_bucket: 5             # Currently filling bucket 5
/// ├─ current_entry: 7              # 7 entries written so far
/// └─ entry_writer: BitWriter       # Writes to index_block_X.bin
/// ```
///
/// # Write Process
///
/// ```text
/// Phase 1: Write entries
/// ┌────────────────────────────────────┐
/// │ [entry0][entry1][entry2]...        │  ← Entries section
/// └────────────────────────────────────┘
///
/// Phase 2: Write bucket array
/// ┌────────────────────────────────────┐
/// │ [0, 0, 2, 5, 5, 8, ...]            │  ← Bucket offsets
/// └────────────────────────────────────┘
///
/// Final file structure:
/// ┌────────────────────────────────────┐
/// │ Entries: [hash,seg,row]...         │
/// ├────────────────────────────────────┤
/// │ Buckets: [offset0, offset1, ...]   │
/// └────────────────────────────────────┘
/// ```
///
/// # Example
///
/// ```rust,ignore
/// let mut builder = IndexBlockBuilder::new(
///     0,      // bucket_start_idx
///     1000,   // bucket_end_idx
///     file_id,
///     index_dir,
/// ).await?;
///
/// // Write entries (must be sorted by hash)
/// for (hash, seg, row) in sorted_entries {
///     let needs_flush = builder.write_entry(hash, seg, row, &metadata);
///     if needs_flush {
///         builder.flush().await?;
///     }
/// }
///
/// // Finalize: writes bucket array and closes file
/// let index_block = builder.build(&metadata).await?;
/// ```
struct IndexBlockBuilder {
    /// First bucket index covered by this index block (inclusive)
    bucket_start_idx: u32,

    /// Last bucket index covered by this index block (exclusive)
    bucket_end_idx: u32,

    /// Offset to the first entry for each bucket in this block.
    /// Length = (bucket_end_idx - bucket_start_idx).
    /// Each value points to the entry position where that bucket's entries start.
    /// Example: [0, 0, 3, 3, 7] means buckets 0-1 are empty, bucket 2 has entries [0,3), etc.
    buckets: Vec<u32>,

    /// Reference to the index file being written
    index_file: MooncakeDataFileRef,

    /// Bit-level writer for variable-bit encoding of entries to disk
    entry_writer: AsyncBitWriter<AsyncFile, AsyncBigEndian>,

    /// The bucket currently being populated with entries.
    /// Increments when entries move to a new bucket.
    current_bucket: u32,

    /// Total number of entries written so far.
    /// Used to populate bucket offsets and track progress.
    current_entry: u32,
}

impl IndexBlockBuilder {
    /// Create a new index block builder.
    ///
    /// Creates a new index file in the specified directory and initializes
    /// the builder for writing entries.
    ///
    /// # Arguments
    ///
    /// * `bucket_start_idx` - First bucket covered by this block
    /// * `bucket_end_idx` - Last bucket (exclusive) covered by this block
    /// * `file_id` - File ID for the new index block file
    /// * `directory` - Directory where index file will be created
    ///
    /// # Returns
    ///
    /// A new `IndexBlockBuilder` ready to accept entries.
    pub async fn new(
        bucket_start_idx: u32,
        bucket_end_idx: u32,
        file_id: u64,
        directory: PathBuf,
    ) -> Result<Self> {
        let file_name = format!("index_block_{}.bin", uuid::Uuid::now_v7());
        let file_path = directory.join(&file_name);

        let file = AsyncFile::create(&file_path).await?;
        let entry_writer = AsyncBitWriter::endian(file, AsyncBigEndian);
        let index_file = create_data_file(file_id, file_path.to_str().unwrap().to_string());

        Ok(Self {
            bucket_start_idx,
            bucket_end_idx,
            buckets: vec![0; (bucket_end_idx - bucket_start_idx) as usize],
            index_file,
            entry_writer,
            current_bucket: bucket_start_idx,
            current_entry: 0,
        })
    }

    /// Write a hash entry to the index block.
    ///
    /// Entries must be written in ascending hash order. The builder automatically
    /// tracks which bucket each entry belongs to.
    ///
    /// # Arguments
    ///
    /// * `hash` - Full 64-bit hash value
    /// * `seg_idx` - File index (which Parquet file)
    /// * `row_idx` - Row index within the file
    /// * `metadata` - Global index metadata for bit widths
    ///
    /// # Returns
    ///
    /// `true` if the internal buffer is full and should be flushed.
    ///
    /// # Panics
    ///
    /// Panics if entries are not written in ascending hash order.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Entries MUST be sorted by hash
    /// let entries = vec![
    ///     (0x0000000000001234, 0, 100),  // hash=0x1234, file=0, row=100
    ///     (0x0000000000005678, 1, 200),  // hash=0x5678, file=1, row=200
    ///     (0x000000000000ABCD, 0, 300),  // hash=0xABCD, file=0, row=300
    /// ];
    ///
    /// for (hash, seg, row) in entries {
    ///     if builder.write_entry(hash, seg, row, &metadata) {
    ///         builder.flush().await?;
    ///     }
    /// }
    /// ```
    ///
    /// # Implementation Details
    ///
    /// ```text
    /// 1. Extract bucket index from upper hash bits
    /// 2. Update bucket offsets for any skipped buckets
    /// 3. Write entry: (lower_hash, seg_idx, row_idx)
    /// 4. Return true if buffer needs flushing
    /// ```
    pub fn write_entry(
        &mut self,
        hash: u64,
        seg_idx: usize,
        row_idx: usize,
        global_index: &GlobalIndex,
    ) -> bool {
        // Extract the bucket index by shifting off the lower hash bits.
        // The upper bits of the hash determine which bucket this entry belongs to.
        // For example, if hash_lower_bits=20, then hash >> 20 gives us the bucket index.
        while (hash >> global_index.hash_lower_bits) != self.current_bucket as u64 {
            // We've skipped to a new bucket that has no entries yet.
            // Update the bucket offset array to record where the current bucket ends.
            // Each skipped bucket's offset points to the current entry index,
            // indicating that the bucket is empty (start == end for that bucket).
            self.current_bucket += 1;
            self.buckets[self.current_bucket as usize] = self.current_entry;
        }

        // Write the lower bits of the hash to the index file.
        // This is the portion of the hash that remains after removing the bucket bits.
        // For example, if hash=0x123456 and hash_lower_bits=20, we write 0x56 (lower 20 bits).
        // The mask (1 << hash_lower_bits) - 1 isolates just the lower bits.
        let _ = self.entry_writer.write(
            global_index.hash_lower_bits,
            hash & ((1 << global_index.hash_lower_bits) - 1),
        );

        // Write the segment index which identifies which data segment contains this row.
        // This is encoded using the specified number of bits (seg_id_bits).
        let _ = self
            .entry_writer
            .write(global_index.seg_id_bits, seg_idx as u32);

        // Write the row index which identifies the specific row within the segment.
        // This is encoded using the specified number of bits (row_id_bits).
        // The write() method returns true if the internal buffer is full and needs flushing.
        let to_flush = self
            .entry_writer
            .write(global_index.row_id_bits, row_idx as u32);

        // Increment the entry counter to track how many entries we've written.
        // This counter is used to update bucket offsets and calculate file positions.
        self.current_entry += 1;

        // Return whether the buffer needs to be flushed to disk.
        // If true, the caller should call flush() to prevent buffer overflow.
        to_flush
    }

    /// Flush buffered entries to disk.
    ///
    /// Should be called when `write_entry` returns `true` to prevent
    /// buffer overflow.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for entry in entries {
    ///     let needs_flush = builder.write_entry(entry.0, entry.1, entry.2, &metadata);
    ///     if needs_flush {
    ///         builder.flush().await?;  // Prevent buffer overflow
    ///     }
    /// }
    /// ```
    pub async fn flush(&mut self) -> Result<()> {
        self.entry_writer.flush().await?;
        Ok(())
    }

    /// Finalize the index block by writing the bucket array and closing the file.
    ///
    /// This method:
    /// 1. Fills in bucket offsets for any remaining empty buckets
    /// 2. Writes the bucket array to the file
    /// 3. Closes the file and creates the `IndexBlock` with mmap
    ///
    /// # File Structure After Build
    ///
    /// ```text
    /// ┌───────────────────────────────────┐
    /// │ Entries Section                   │
    /// │ [(hash,seg,row), (hash,seg,row)]..│ ← Written during write_entry
    /// ├───────────────────────────────────┤
    /// │ Bucket Array                      │
    /// │ [offset0][offset1][offset2]..     │ ← Written during build
    /// └───────────────────────────────────┘
    /// ```
    ///
    /// # Arguments
    ///
    /// * `metadata` - Global index metadata for bucket bit width
    ///
    /// # Returns
    ///
    /// An `IndexBlock` with the file memory-mapped for reading.
    pub async fn build(mut self, metadata: &GlobalIndex) -> Result<IndexBlock> {
        for i in self.current_bucket + 1..self.bucket_end_idx {
            self.buckets[i as usize] = self.current_entry;
        }
        let bucket_start_offset = (self.current_entry as u64)
            * (metadata.hash_lower_bits + metadata.seg_id_bits + metadata.row_id_bits) as u64;
        let buckets = std::mem::take(&mut self.buckets);
        for cur_bucket in buckets {
            let to_flush = self.entry_writer.write(metadata.bucket_bits, cur_bucket);
            if to_flush {
                self.entry_writer.flush().await?;
            }
        }
        self.entry_writer.close().await?;

        Ok(IndexBlock::new(
            self.bucket_start_idx,
            self.bucket_end_idx,
            bucket_start_offset,
            self.index_file,
        )
        .await)
    }
}

/// Builder for creating global indices from various sources.
///
/// Supports building indices from:
/// - **Flush**: New data being written to disk
/// - **Merge**: Combining multiple existing indices
/// - **Compaction**: Merging indices with row filtering/remapping
///
/// # Example: Build from Flush
///
/// ```rust,ignore
/// let entries = vec![
///     (key1, file_idx, row_idx),
///     (key2, file_idx, row_idx),
/// ];
///
/// let index = GlobalIndexBuilder::new()
///     .set_files(vec![parquet_file])
///     .set_directory(index_dir)
///     .build_from_flush(entries, file_id)
///     .await?;
/// ```
///
/// # Example: Build from Merge
///
/// ```rust,ignore
/// let merged = GlobalIndexBuilder::new()
///     .set_directory(index_dir)
///     .build_from_merge(vec![index1, index2, index3].into(), file_id)
///     .await?;
/// ```
pub struct GlobalIndexBuilder {
    num_rows: u32,
    files: Vec<MooncakeDataFileRef>,
    directory: PathBuf,
}

impl Default for GlobalIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalIndexBuilder {
    /// Create a new index builder.
    ///
    /// Use `set_files` and `set_directory` to configure before building.
    pub fn new() -> Self {
        Self {
            num_rows: 0,
            files: vec![],
            directory: PathBuf::new(),
        }
    }

    /// Set the directory where index block files will be created.
    ///
    /// # Arguments
    ///
    /// * `directory` - Path to directory for index files
    pub fn set_directory(&mut self, directory: PathBuf) -> &mut Self {
        self.directory = directory;
        self
    }

    /// Set the Parquet data files this index will reference.
    ///
    /// # Arguments
    ///
    /// * `files` - Vector of data file references to be indexed
    pub fn set_files(&mut self, files: Vec<MooncakeDataFileRef>) -> &mut Self {
        self.files = files;
        self
    }

    /// Create a global index metadata structure with optimized bit allocations.
    ///
    /// Calculates optimal bit widths based on data size:
    /// - `bucket_bits`: Based on total rows
    /// - `seg_id_bits`: Based on number of files
    /// - `hash_upper_bits`: For bucket indexing
    /// - `hash_lower_bits`: For hash verification
    ///
    /// # Returns
    ///
    /// Tuple of (num_buckets, GlobalIndex) with calculated metadata.
    fn create_global_index(&mut self) -> (u32, GlobalIndex) {
        let num_rows = self.num_rows;
        let bucket_bits = 32 - num_rows.leading_zeros();
        let num_buckets = (num_rows / 4 + 2).next_power_of_two();
        let upper_bits = num_buckets.trailing_zeros();
        let lower_bits = 64 - upper_bits;
        let seg_id_bits = 32 - (self.files.len() as u32).trailing_zeros();
        let global_index = GlobalIndex {
            files: std::mem::take(&mut self.files),
            num_rows,
            hash_bits: HASH_BITS,
            hash_upper_bits: upper_bits,
            hash_lower_bits: lower_bits,
            seg_id_bits,
            row_id_bits: 32,
            bucket_bits,
            index_blocks: vec![],
        };
        (num_buckets, global_index)
    }

    /// Create file ID remapping tables for merging multiple indices.
    ///
    /// When merging indices, each index may reference overlapping file IDs.
    /// This function creates remapping tables to assign unique sequential
    /// file IDs in the merged index.
    ///
    /// # Arguments
    ///
    /// * `file_indice_iter` - Iterator over indices to merge
    ///
    /// # Returns
    ///
    /// Vector of remapping tables, one per input index.
    ///
    /// # Example
    ///
    /// ```text
    /// Index 1: files [0, 1, 2]
    /// Index 2: files [0, 1]
    /// Index 3: files [0, 1, 2, 3]
    ///
    /// Remapping:
    ///   Index 1: [0→0, 1→1, 2→2]
    ///   Index 2: [0→3, 1→4]
    ///   Index 3: [0→5, 1→6, 2→7, 3→8]
    ///
    /// Merged index has files [0, 1, 2, 3, 4, 5, 6, 7, 8]
    /// ```
    fn create_file_id_remap_at_merge<'a>(
        file_indice_iter: impl Iterator<Item = &'a GlobalIndex>,
    ) -> Vec<Vec<u32>> {
        let mut file_id_remaps = vec![];
        let mut file_id_after_remap = 0;
        for index in file_indice_iter {
            let mut file_id_remap = vec![INVALID_FILE_ID; index.files.len()];
            for (_, item) in file_id_remap.iter_mut().enumerate().take(index.files.len()) {
                *item = file_id_after_remap;
                file_id_after_remap += 1;
            }
            file_id_remaps.push(file_id_remap);
        }
        file_id_remaps
    }

    // ================================
    // Build from flush
    // ================================

    /// Build an index from newly flushed data.
    ///
    /// Used when writing new data to Parquet files - creates an index
    /// for the primary keys in the new data.
    ///
    /// # Arguments
    ///
    /// * `entries` - Vector of (key, file_idx, row_idx) tuples
    /// * `file_id` - File ID for the new index block file
    ///
    /// # Returns
    ///
    /// A new `GlobalIndex` covering the provided entries.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Just wrote rows to file index 0
    /// let entries = vec![
    ///     (12345, 0, 0),  // key=12345, file=0, row=0
    ///     (67890, 0, 1),  // key=67890, file=0, row=1
    ///     (11111, 0, 2),  // key=11111, file=0, row=2
    /// ];
    ///
    /// let index = GlobalIndexBuilder::new()
    ///     .set_files(vec![parquet_file])
    ///     .set_directory(index_dir)
    ///     .build_from_flush(entries, file_id)
    ///     .await?;
    /// ```
    ///
    /// # Implementation
    ///
    /// 1. Hashes all keys using splitmix64
    /// 2. Sorts entries by hash
    /// 3. Writes entries to index block file
    /// 4. Creates bucket array pointing to entries
    pub async fn build_from_flush(
        mut self,
        mut entries: Vec<(u64, usize, usize)>,
        file_id: u64,
    ) -> Result<GlobalIndex> {
        self.num_rows = entries.len() as u32;
        for entry in &mut entries {
            entry.0 = splitmix64(entry.0);
        }
        entries.sort_unstable_by_key(|entry| entry.0);
        let global_index = self.build(entries.into_iter(), file_id).await?;
        Ok(global_index)
    }

    /// Internal build method that creates an index from an iterator of entries.
    ///
    /// # Arguments
    ///
    /// * `iter` - Iterator yielding (hash, seg_idx, row_idx) tuples (must be sorted by hash)
    /// * `file_id` - File ID for the index block file
    ///
    /// # Returns
    ///
    /// A new `GlobalIndex` with a single index block.
    async fn build(
        mut self,
        iter: impl Iterator<Item = (u64, usize, usize)>,
        file_id: u64,
    ) -> Result<GlobalIndex> {
        let (num_buckets, mut global_index) = self.create_global_index();
        let mut index_blocks = Vec::new();
        let mut index_block_builder =
            IndexBlockBuilder::new(0, num_buckets + 1, file_id, self.directory.clone()).await?;
        for entry in iter {
            let to_flush =
                index_block_builder.write_entry(entry.0, entry.1, entry.2, &global_index);
            if to_flush {
                index_block_builder.flush().await?;
            }
        }
        index_blocks.push(index_block_builder.build(&global_index).await?);
        global_index.index_blocks = index_blocks;
        Ok(global_index)
    }

    // ================================
    // Build from merge
    // ================================

    /// Merge multiple indices into a single consolidated index.
    ///
    /// Combines hash entries from multiple indices while remapping file IDs
    /// to avoid conflicts.
    ///
    /// # Arguments
    ///
    /// * `indices` - Set of indices to merge
    /// * `file_id` - File ID for the new merged index block file
    ///
    /// # Returns
    ///
    /// A new `GlobalIndex` containing all entries from input indices.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Merge 3 indices into one
    /// let indices = HashSet::from([index1, index2, index3]);
    ///
    /// let merged = GlobalIndexBuilder::new()
    ///     .set_directory(index_dir)
    ///     .build_from_merge(indices, file_id)
    ///     .await?;
    ///
    /// // merged.files contains all files from all 3 indices
    /// // merged.index_blocks[0] contains all hash entries
    /// ```
    ///
    /// # Performance
    ///
    /// Uses a min-heap to efficiently merge sorted entries from multiple
    /// indices in O(N log K) time where N = total entries, K = number of indices.
    #[allow(clippy::mutable_key_type)]
    pub async fn build_from_merge(
        mut self,
        indices: HashSet<GlobalIndex>,
        file_id: u64,
    ) -> Result<GlobalIndex> {
        self.num_rows = indices.iter().map(|index| index.num_rows).sum();
        self.files = indices
            .iter()
            .flat_map(|index| index.files.clone())
            .collect();
        let file_id_remaps = Self::create_file_id_remap_at_merge(indices.iter());
        let mut iters = Vec::with_capacity(indices.len());
        for (idx, index) in indices.iter().enumerate() {
            iters.push(index.create_iterator(&file_id_remaps[idx]));
        }
        let merge_iter = GlobalIndexMergingIterator::new(iters);
        self.build_from_merging_iterator(merge_iter, file_id).await
    }

    /// Build an index from a merging iterator.
    ///
    /// Internal method used by merge operations to create an index from
    /// a sorted stream of entries.
    ///
    /// # Arguments
    ///
    /// * `iter` - Merging iterator that yields sorted entries
    /// * `file_id` - File ID for the index block file
    async fn build_from_merging_iterator(
        mut self,
        mut iter: GlobalIndexMergingIterator<'_>,
        file_id: u64,
    ) -> Result<GlobalIndex> {
        let (num_buckets, mut global_index) = self.create_global_index();
        let mut index_block_builder =
            IndexBlockBuilder::new(0, num_buckets + 1, file_id, self.directory.clone()).await?;
        while let Some(entry) = iter.next() {
            let to_flush =
                index_block_builder.write_entry(entry.0, entry.1, entry.2, &global_index);
            if to_flush {
                index_block_builder.flush().await?;
            }
        }

        let mut index_blocks = Vec::new();
        index_blocks.push(index_block_builder.build(&global_index).await?);
        global_index.index_blocks = index_blocks;
        Ok(global_index)
    }

    // ================================
    // Build from merge with predicate
    // ================================

    /// Build an index for compaction with row filtering and remapping.
    ///
    /// Unlike [`build_from_merge`], this only includes entries that pass a predicate
    /// and remaps their locations to reflect compacted file structure.
    ///
    /// Used during table compaction when rows are being reorganized, deleted, or
    /// moved between files.
    ///
    /// # Arguments
    ///
    /// * `num_rows` - Number of rows after compaction (after filtering)
    /// * `file_id` - File ID for the new index block file
    /// * `indices` - Indices to merge from pre-compaction files
    /// * `new_data_files` - Post-compaction data files
    /// * `get_remapped_record_location` - Maps old locations to new locations (or None if deleted)
    /// * `get_seg_idx` - Extracts segment index from a record location
    ///
    /// # Returns
    ///
    /// A new `GlobalIndex` referencing `new_data_files` with remapped row locations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Compaction deleted some rows and moved others
    /// let location_map = HashMap::new();  // old_loc -> new_loc
    ///
    /// let compacted_index = GlobalIndexBuilder::new()
    ///     .set_directory(index_dir)
    ///     .build_from_merge_for_compaction(
    ///         new_row_count,
    ///         file_id,
    ///         old_indices,
    ///         new_data_files,
    ///         |old_loc| location_map.get(&old_loc).copied(),  // None = deleted
    ///         |new_loc| match new_loc {
    ///             RecordLocation::DiskFile(file_id, _) => file_to_seg[&file_id],
    ///             _ => panic!()
    ///         },
    ///     )
    ///     .await?;
    /// ```
    pub async fn build_from_merge_for_compaction<GetRemappedRecLoc, GetSegIdx>(
        mut self,
        num_rows: u32,
        file_id: u64,
        indices: Vec<GlobalIndex>,
        new_data_files: Vec<MooncakeDataFileRef>,
        get_remapped_record_location: GetRemappedRecLoc,
        get_seg_idx: GetSegIdx,
    ) -> Result<GlobalIndex>
    where
        GetRemappedRecLoc: FnMut(RecordLocation) -> Option<RecordLocation>,
        GetSegIdx: FnMut(RecordLocation) -> usize, /*seg_idx*/
    {
        // Assign data files before compaction, used to compose old record location and look it up with [`get_remapped_record_location`] and new record location after compaction.
        self.files = indices
            .iter()
            .flat_map(|index| index.files.clone())
            .collect();
        self.num_rows = num_rows;

        let file_id_remaps = Self::create_file_id_remap_at_merge(indices.iter());
        let mut iters = Vec::with_capacity(indices.len());
        for (idx, index) in indices.iter().enumerate() {
            iters.push(index.create_iterator(&file_id_remaps[idx]));
        }
        let merge_iter = GlobalIndexMergingIterator::new(iters);
        self.build_from_merging_iterator_with_predicate(
            file_id,
            merge_iter,
            new_data_files,
            get_remapped_record_location,
            get_seg_idx,
        )
        .await
    }

    /// Internal method to build an index with location remapping.
    ///
    /// Iterates through merged entries, applies the remapping predicate,
    /// and builds a new index referencing the new data files.
    ///
    /// # Arguments
    ///
    /// * `file_id` - File ID for the index block file
    /// * `iter` - Merging iterator over old entries
    /// * `new_data_files` - Post-compaction data files
    /// * `get_remapped_record_location` - Maps old → new locations (None = deleted)
    /// * `get_seg_idx` - Extracts file index from location
    async fn build_from_merging_iterator_with_predicate<GetRemappedRecLoc, GetSegIdx>(
        mut self,
        file_id: u64,
        mut iter: GlobalIndexMergingIterator<'_>,
        new_data_files: Vec<MooncakeDataFileRef>,
        mut get_remapped_record_location: GetRemappedRecLoc,
        mut get_seg_idx: GetSegIdx,
    ) -> Result<GlobalIndex>
    where
        GetRemappedRecLoc: FnMut(RecordLocation) -> Option<RecordLocation>,
        GetSegIdx: FnMut(RecordLocation) -> usize, /*seg_idx*/
    {
        let (num_buckets, mut global_index) = self.create_global_index();
        let mut index_block_builder =
            IndexBlockBuilder::new(0, num_buckets + 1, file_id, self.directory.clone()).await?;

        while let Some((hash, old_seg_idx, old_row_idx)) = iter.next() {
            let old_record_location =
                RecordLocation::DiskFile(global_index.files[old_seg_idx].file_id(), old_row_idx);
            if let Some(new_record_location) = get_remapped_record_location(old_record_location) {
                let new_row_idx = match new_record_location {
                    RecordLocation::DiskFile(_, offset) => offset,
                    _ => panic!("Expected DiskFile variant"),
                };
                let new_seg_idx = get_seg_idx(new_record_location);
                let to_flush =
                    index_block_builder.write_entry(hash, new_seg_idx, new_row_idx, &global_index);
                if to_flush {
                    index_block_builder.flush().await?;
                }
            }
            // The record doesn't exist in compacted data files, which means the corresponding row doesn't exist in the data file after compaction, simply ignore.
        }

        let mut index_blocks = Vec::new();
        index_blocks.push(index_block_builder.build(&global_index).await?);
        global_index.index_blocks = index_blocks;

        // Now all the (hash, seg_idx, row_idx) points to the new files passed in.
        global_index.files = new_data_files;

        Ok(global_index)
    }
}

// ================================
// Iterators for merging indices
// ================================

/// Iterator over all hash entries in a single index block.
///
/// Sequentially reads entries from an index block, applying file ID
/// remapping during iteration.
///
/// # Structure
///
/// ```text
/// IndexBlockIterator
/// ├─ collection: &IndexBlock        # Index block being iterated
/// ├─ current_bucket: 5              # Currently reading bucket 5
/// ├─ current_entry: 50              # On entry 50
/// ├─ bucket_reader: BitReader       # Reads bucket offsets
/// ├─ entry_reader: BitReader        # Reads hash entries
/// └─ file_id_remap: [0→0, 1→3]     # Remap file IDs during iteration
/// ```
struct IndexBlockIterator<'a> {
    collection: &'a IndexBlock,
    metadata: &'a GlobalIndex,
    current_bucket: u32,
    current_bucket_entry_end: u32,
    current_entry: u32,
    current_upper_hash: u64,
    bucket_reader: BitReader<Cursor<&'a [u8]>, BigEndian>,
    entry_reader: BitReader<Cursor<&'a [u8]>, BigEndian>,
    file_id_remap: &'a Vec<u32>,
}

impl<'a> IndexBlockIterator<'a> {
    /// Create a new iterator over an index block.
    ///
    /// Initializes readers and reads the first bucket's entry range.
    ///
    /// # Arguments
    ///
    /// * `collection` - The index block to iterate over
    /// * `metadata` - Global index metadata for bit widths
    /// * `file_id_remap` - Mapping from old file IDs to new file IDs
    fn new(
        collection: &'a IndexBlock,
        global_index: &'a GlobalIndex,
        file_id_remap: &'a Vec<u32>,
    ) -> Self {
        let mut bucket_reader = BitReader::endian(
            Cursor::new(collection.data.as_ref().as_ref().unwrap().as_ref()),
            BigEndian,
        );
        let entry_reader = bucket_reader.clone();
        bucket_reader
            .seek_bits(SeekFrom::Start(collection.bucket_start_offset))
            .unwrap();
        let _ = bucket_reader
            .read_unsigned_var::<u32>(global_index.bucket_bits)
            .unwrap();
        let current_bucket_entry_end = bucket_reader
            .read_unsigned_var::<u32>(global_index.bucket_bits)
            .unwrap();
        Self {
            collection,
            metadata: global_index,
            bucket_reader,
            entry_reader,
            current_bucket: collection.bucket_start_idx,
            current_bucket_entry_end,
            current_entry: 0,
            current_upper_hash: 0,
            file_id_remap,
        }
    }

    /// Get the next hash entry from the index block.
    ///
    /// Advances through entries bucket by bucket, skipping empty buckets.
    /// Applies file ID remapping to convert old file indices to new ones.
    ///
    /// # Returns
    ///
    /// * `Some((hash, remapped_seg_idx, row_idx))` - Next entry with remapped file ID
    /// * `None` - All entries exhausted
    fn next(
        &mut self,
    ) -> Option<(
        u64,   /*hash*/
        usize, /*seg_idx*/
        usize, /*row_idx*/
    )> {
        if self.current_bucket == self.collection.bucket_end_idx - 1 {
            return None;
        }
        while self.current_entry == self.current_bucket_entry_end {
            self.current_bucket += 1;
            if self.current_bucket == self.collection.bucket_end_idx - 1 {
                return None;
            }
            self.current_bucket_entry_end = self
                .bucket_reader
                .read_unsigned_var::<u32>(self.metadata.bucket_bits)
                .unwrap();
            self.current_upper_hash += 1 << self.metadata.hash_lower_bits;
        }
        let (lower_hash, seg_idx, row_idx) = self
            .collection
            .read_entry(&mut self.entry_reader, self.metadata);
        self.current_entry += 1;
        let seg_idx = self.file_id_remap.get(seg_idx).unwrap();
        assert_ne!(*seg_idx, INVALID_FILE_ID);
        Some((
            lower_hash + self.current_upper_hash,
            *seg_idx as usize,
            row_idx,
        ))
    }
}

/// Iterator over all hash entries across all blocks in a global index.
///
/// Wraps multiple `IndexBlockIterator`s and seamlessly transitions
/// between index blocks.
///
/// # Example
///
/// ```rust,ignore
/// let identity_remap = (0..index.files.len() as u32).collect();
/// let mut iter = index.create_iterator(&identity_remap);
///
/// while let Some((hash, seg_idx, row_idx)) = iter.next() {
///     println!("Entry: hash={:x}, file={}, row={}", hash, seg_idx, row_idx);
/// }
/// ```
pub struct GlobalIndexIterator<'a> {
    index: &'a GlobalIndex,
    block_idx: usize,
    block_iter: Option<IndexBlockIterator<'a>>,
    file_id_remap: &'a Vec<u32>,
}

impl<'a> GlobalIndexIterator<'a> {
    /// Create a new iterator over all blocks in a global index.
    ///
    /// # Arguments
    ///
    /// * `index` - The global index to iterate over
    /// * `file_id_remap` - Mapping from old file IDs to new file IDs
    pub fn new(index: &'a GlobalIndex, file_id_remap: &'a Vec<u32>) -> Self {
        let mut block_iter = None;
        let block_idx = 0;
        if !index.index_blocks.is_empty() {
            block_iter = Some(index.index_blocks[0].create_iterator(index, file_id_remap));
        }
        Self {
            index,
            block_idx,
            block_iter,
            file_id_remap,
        }
    }

    /// Get the next hash entry from the global index.
    ///
    /// Automatically transitions between index blocks as they are exhausted.
    ///
    /// # Returns
    ///
    /// * `Some((hash, seg_idx, row_idx))` - Next entry from any block
    /// * `None` - All blocks exhausted
    pub fn next(
        &mut self,
    ) -> Option<(
        u64,   /*hash*/
        usize, /*seg_idx*/
        usize, /*row_idx*/
    )> {
        loop {
            if let Some(ref mut iter) = self.block_iter {
                if let Some(item) = iter.next() {
                    return Some(item);
                }
            }
            self.block_idx += 1;
            if self.block_idx >= self.index.index_blocks.len() {
                return None;
            }
            self.block_iter = Some(
                self.index.index_blocks[self.block_idx]
                    .create_iterator(self.index, self.file_id_remap),
            );
        }
    }
}

/// Min-heap based iterator for merging multiple sorted index iterators.
///
/// Efficiently merges K sorted streams of hash entries into a single
/// sorted stream using a min-heap.
///
/// # Algorithm
///
/// ```text
/// Input: 3 sorted index iterators
///   Iter1: [100, 250, 500, ...]
///   Iter2: [150, 300, 400, ...]
///   Iter3: [120, 200, 350, ...]
///
/// Heap (min-heap by hash):
///   Step 1: [100(Iter1), 120(Iter3), 150(Iter2)]  → Pop 100
///   Step 2: [120(Iter3), 150(Iter2), 250(Iter1)]  → Pop 120
///   Step 3: [150(Iter2), 200(Iter3), 250(Iter1)]  → Pop 150
///   ...
///
/// Output: [100, 120, 150, 200, 250, 300, 350, 400, 500, ...]
/// ```
///
/// # Complexity
///
/// - Time: O(N log K) where N = total entries, K = number of iterators
/// - Space: O(K) for the heap
///
/// # Example
///
/// ```rust,ignore
/// let iters = indices.iter()
///     .map(|idx| idx.create_iterator(&file_id_remap))
///     .collect();
///
/// let mut merge_iter = GlobalIndexMergingIterator::new(iters);
///
/// while let Some((hash, seg, row)) = merge_iter.next() {
///     // Process merged entries in sorted order
/// }
/// ```
pub struct GlobalIndexMergingIterator<'a> {
    heap: BinaryHeap<HeapItem<'a>>,
}

/// Heap entry for merging iterator, containing a value and its source iterator.
///
/// Ordered by hash value for min-heap construction.
struct HeapItem<'a> {
    value: (
        u64,   /*hash*/
        usize, /*seg_idx*/
        usize, /*row_idx*/
    ),
    iter: GlobalIndexIterator<'a>,
}

impl PartialEq for HeapItem<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.value.0 == other.value.0
    }
}
impl Eq for HeapItem<'_> {}

impl PartialOrd for HeapItem<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap
        other.value.0.cmp(&self.value.0)
    }
}

impl<'a> GlobalIndexMergingIterator<'a> {
    /// Create a new merging iterator from multiple index iterators.
    ///
    /// Initializes a min-heap with the first entry from each iterator.
    ///
    /// # Arguments
    ///
    /// * `iterators` - Vector of index iterators to merge
    ///
    /// # Returns
    ///
    /// A merging iterator that yields entries in sorted order by hash.
    pub fn new(iterators: Vec<GlobalIndexIterator<'a>>) -> Self {
        let mut heap = BinaryHeap::new();
        for mut it in iterators {
            if let Some(value) = it.next() {
                heap.push(HeapItem { value, iter: it });
            }
        }
        Self { heap }
    }

    /// Get the next entry in sorted order across all iterators.
    ///
    /// Pops the minimum hash from the heap, returns it, and pushes the
    /// next entry from that iterator back into the heap.
    ///
    /// # Returns
    ///
    /// * `Some((hash, seg_idx, row_idx))` - Next entry in sorted order
    /// * `None` - All iterators exhausted
    pub fn next(&mut self) -> Option<(u64, usize, usize)> {
        if let Some(mut heap_item) = self.heap.pop() {
            let result = heap_item.value;
            if let Some(next_value) = heap_item.iter.next() {
                self.heap.push(HeapItem {
                    value: next_value,
                    iter: heap_item.iter,
                });
            }
            Some(result)
        } else {
            None
        }
    }
}

// ================================
// Debug Helpers
// ================================
impl IndexBlock {
    /// Format index block for debugging output.
    ///
    /// Displays bucket offsets and hash entries in human-readable form.
    ///
    /// # Arguments
    ///
    /// * `f` - Formatter to write to
    /// * `metadata` - Global index metadata for decoding entries
    fn fmt(&self, f: &mut fmt::Formatter<'_>, metadata: &GlobalIndex) -> fmt::Result {
        write!(
            f,
            "\nIndexBlock {{ \n   bucket_start_idx: {}, \n   bucket_end_idx: {},",
            self.bucket_start_idx, self.bucket_end_idx
        )?;
        let cursor = Cursor::new(self.data.as_ref().as_ref().unwrap().as_ref());
        let mut reader = BitReader::endian(cursor, BigEndian);
        write!(f, "\n   Buckets: ")?;
        let mut num = 0;
        reader
            .seek_bits(SeekFrom::Start(self.bucket_start_offset))
            .unwrap();
        for _i in 0..self.bucket_end_idx {
            num = reader
                .read_unsigned_var::<u32>(metadata.bucket_bits)
                .unwrap();
            write!(f, "{num} ")?;
        }
        write!(f, "\n   Entries: ")?;
        reader.seek_bits(SeekFrom::Start(0)).unwrap();
        for _i in 0..num {
            let (hash, seg_idx, row_idx) = self.read_entry(&mut reader, metadata);
            write!(f, "\n     {hash} {seg_idx} {row_idx}")?;
        }
        write!(f, "\n}}")?;
        Ok(())
    }
}

impl Debug for GlobalIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GlobalIndex {{ files: {:?}, num_rows: {}, hash_bits: {}, hash_upper_bits: {}, hash_lower_bits: {}, seg_id_bits: {}, row_id_bits: {}, bucket_bits: {} ", self.files, self.num_rows, self.hash_bits, self.hash_upper_bits, self.hash_lower_bits, self.seg_id_bits, self.row_id_bits, self.bucket_bits)?;
        for block in &self.index_blocks {
            block.fmt(f, self)?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
/// Test helper to prepare hashes for lookup operations.
///
/// Wrapper around `GlobalIndex::prepare_hashes_for_lookup` for testing.
///
/// # Arguments
///
/// * `values` - Slice of keys to hash and prepare
///
/// # Returns
///
/// Vector of (key, hash) pairs sorted by hash.
pub fn test_get_hashes_for_index(values: &[u64]) -> Vec<(u64, u64)> {
    GlobalIndex::hash_sort_and_dedup(values.iter().copied())
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use tracing::debug;

    use crate::storage::storage_utils::{create_data_file, FileId};

    #[tokio::test]
    async fn test_new() {
        let data_file = create_data_file(/*file_id=*/ 0, "a.parquet".to_string());
        let files = vec![data_file.clone()];
        let hash_entries = vec![
            (1, 0, 0),
            (2, 0, 1),
            (3, 0, 2),
            (4, 0, 3),
            (5, 0, 4),
            (16, 0, 5),
            (214141, 0, 6),
            (2141, 0, 7),
            (21141, 0, 8),
            (219511, 0, 9),
            (1421141, 0, 10),
            (1111111141, 0, 11),
            (99999, 0, 12),
        ];
        let mut builder = GlobalIndexBuilder::new();
        builder
            .set_files(files)
            .set_directory(tempfile::tempdir().unwrap().keep());
        let index = builder
            .build_from_flush(hash_entries.clone(), /*file_id=*/ 1)
            .await
            .unwrap();

        // Search for a non-existent key doesn't panic.
        assert!(index
            .search_values(&test_get_hashes_for_index(&[0]))
            .await
            .is_empty());

        let data_file_ids = [data_file.file_id()];
        for (hash, seg_idx, row_idx) in hash_entries.iter() {
            let expected_record_loc = RecordLocation::DiskFile(data_file_ids[*seg_idx], *row_idx);
            assert_eq!(
                index
                    .search_values(&test_get_hashes_for_index(&[*hash]))
                    .await,
                vec![(*hash, expected_record_loc)]
            );
        }

        let mut hash_entry_num = 0;
        let file_id_remap = vec![0; index.files.len()];
        for block in index.index_blocks.iter() {
            let mut index_block_iter = block.create_iterator(&index, &file_id_remap);
            while let Some((hash, seg_idx, row_idx)) = index_block_iter.next() {
                debug!(?hash, seg_idx, row_idx, "index entry");
                hash_entry_num += 1;
            }
        }
        // Check all hash entries are stored and iterated through via index iterator.
        assert_eq!(hash_entry_num, hash_entries.len());
    }

    #[tokio::test]
    async fn test_merge() {
        let files = vec![
            create_data_file(/*file_id=*/ 1, "1.parquet".to_string()),
            create_data_file(/*file_id=*/ 2, "2.parquet".to_string()),
            create_data_file(/*file_id=*/ 3, "3.parquet".to_string()),
        ];
        let vec = (0..100).map(|i| (i as u64, i % 3, i)).collect::<Vec<_>>();
        let mut builder = GlobalIndexBuilder::new();
        builder
            .set_files(files)
            .set_directory(tempfile::tempdir().unwrap().keep());
        let index1 = builder.build_from_flush(vec, /*file_id=*/ 4).await.unwrap();

        let files = vec![
            create_data_file(/*file_id=*/ 5, "4.parquet".to_string()),
            create_data_file(/*file_id=*/ 6, "5.parquet".to_string()),
        ];
        let vec = (100..200).map(|i| (i as u64, i % 2, i)).collect::<Vec<_>>();
        let mut builder = GlobalIndexBuilder::new();
        builder
            .set_files(files)
            .set_directory(tempfile::tempdir().unwrap().keep());
        let index2 = builder.build_from_flush(vec, /*file_id=*/ 7).await.unwrap();

        let mut builder = GlobalIndexBuilder::new();
        builder.set_directory(tempfile::tempdir().unwrap().keep());
        let merged = builder
            .build_from_merge(
                HashSet::<GlobalIndex>::from([index1, index2]),
                /*file_id=*/ 8,
            )
            .await
            .unwrap();

        let values = (0..200).collect::<Vec<_>>();
        let mut ret = merged
            .search_values(&test_get_hashes_for_index(&values))
            .await;
        ret.sort_by_key(|(value, _)| *value);
        assert_eq!(ret.len(), 200);
        for (value, pos) in ret.iter() {
            let RecordLocation::DiskFile(FileId(file_id), _) = pos else {
                panic!("No record location found for {value}");
            };
            // Check for the first file indice.
            // The second batch of data file ids starts with 1.
            if *value < 100 {
                assert_eq!(*file_id, *value % 3 + 1);
            }
            // Check for the second file indice.
            // The second batch of data file ids starts with 5.
            else {
                assert_eq!(*file_id, (*value - 100) % 2 + 5);
            }
        }
    }
}
