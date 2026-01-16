//! Index entry encoding and decoding.
//!
//! This module defines the entry format and encoding/decoding logic for
//! individual hash entries in the index.

use bytes::{Buf, BufMut, Bytes, BytesMut};

use super::error::{S3IndexError, S3IndexResult};
use super::format::S3IndexHeader;

/// A single entry in the S3 index.
///
/// Each entry maps a hash value to a location in a data file.
///
/// # Fields
///
/// * `lower_hash` - Lower bits of the key's hash (upper bits determine bucket)
/// * `file_idx` - Index into the file list (which data file)
/// * `row_idx` - Row number within the data file
///
/// # Example
///
/// ```rust,ignore
/// // Entry for user_id=12345 in file[2], row 892
/// S3IndexEntry {
///     lower_hash: 0x5E6F7890,  // Lower 32 bits of hash(12345)
///     file_idx: 2,
///     row_idx: 892,
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S3IndexEntry {
    /// Lower bits of the hash (for verification during lookup).
    pub lower_hash: u64,

    /// Index into the file list.
    pub file_idx: u32,

    /// Row index within the file.
    pub row_idx: u64,
}

impl S3IndexEntry {
    /// Create a new entry.
    pub fn new(lower_hash: u64, file_idx: u32, row_idx: u64) -> Self {
        Self {
            lower_hash,
            file_idx,
            row_idx,
        }
    }

    /// Encode a single entry to bytes.
    ///
    /// Uses fixed-width encoding based on the header's bit widths.
    /// The entry is packed into the minimum number of bytes needed.
    pub fn encode(&self, header: &S3IndexHeader) -> Bytes {
        let mut buf = BytesMut::with_capacity(header.entry_size as usize);

        // Pack bits: lower_hash | file_idx | row_idx
        let total_bits = header.hash_lower_bits + header.seg_id_bits + header.row_id_bits;
        let total_bytes = (total_bits + 7) / 8;

        // Create a u128 to hold all bits, then extract bytes
        let mut packed: u128 = 0;
        let mut bit_pos = 0u32;

        // Add row_idx (least significant)
        packed |= (self.row_idx as u128) & ((1u128 << header.row_id_bits) - 1);
        bit_pos += header.row_id_bits;

        // Add file_idx
        packed |= ((self.file_idx as u128) & ((1u128 << header.seg_id_bits) - 1)) << bit_pos;
        bit_pos += header.seg_id_bits;

        // Add lower_hash (most significant of the packed data)
        packed |= ((self.lower_hash as u128) & ((1u128 << header.hash_lower_bits) - 1)) << bit_pos;

        // Write bytes in big-endian order
        let bytes = packed.to_be_bytes();
        let start = 16 - total_bytes as usize;
        buf.put_slice(&bytes[start..]);

        buf.freeze()
    }

    /// Decode a single entry from bytes.
    pub fn decode(data: &[u8], header: &S3IndexHeader) -> S3IndexResult<Self> {
        let entry_size = header.entry_size as usize;
        if data.len() < entry_size {
            return Err(S3IndexError::EntryDecodeError {
                message: format!(
                    "Entry data too small: {} bytes, expected {}",
                    data.len(),
                    entry_size
                ),
            });
        }

        // Read bytes into u128 (big-endian)
        let mut packed: u128 = 0;
        for &byte in &data[..entry_size] {
            packed = (packed << 8) | (byte as u128);
        }

        // Extract fields (in reverse order of encoding)
        let row_mask = (1u128 << header.row_id_bits) - 1;
        let row_idx = (packed & row_mask) as u64;
        packed >>= header.row_id_bits;

        let seg_mask = (1u128 << header.seg_id_bits) - 1;
        let file_idx = (packed & seg_mask) as u32;
        packed >>= header.seg_id_bits;

        let hash_mask = (1u128 << header.hash_lower_bits) - 1;
        let lower_hash = (packed & hash_mask) as u64;

        Ok(Self {
            lower_hash,
            file_idx,
            row_idx,
        })
    }

    /// Encode multiple entries to bytes.
    pub fn encode_all(entries: &[S3IndexEntry], header: &S3IndexHeader) -> Bytes {
        let entry_size = header.entry_size as usize;
        let mut buf = BytesMut::with_capacity(entries.len() * entry_size);

        for entry in entries {
            buf.put(entry.encode(header));
        }

        buf.freeze()
    }

    /// Decode multiple entries from bytes.
    pub fn decode_all(data: &[u8], header: &S3IndexHeader) -> S3IndexResult<Vec<S3IndexEntry>> {
        let entry_size = header.entry_size as usize;
        if entry_size == 0 {
            return Ok(Vec::new());
        }

        let num_entries = data.len() / entry_size;
        let mut entries = Vec::with_capacity(num_entries);

        for i in 0..num_entries {
            let start = i * entry_size;
            let end = start + entry_size;
            entries.push(Self::decode(&data[start..end], header)?);
        }

        Ok(entries)
    }
}

/// Reconstruct the full 64-bit hash from bucket index and lower hash.
///
/// # Arguments
///
/// * `bucket_idx` - The bucket index (derived from upper hash bits)
/// * `lower_hash` - The lower hash bits stored in the entry
/// * `hash_lower_bits` - Number of bits in lower_hash
///
/// # Returns
///
/// The reconstructed 64-bit hash value.
pub fn reconstruct_hash(bucket_idx: u32, lower_hash: u64, hash_lower_bits: u32) -> u64 {
    ((bucket_idx as u64) << hash_lower_bits) | lower_hash
}

/// Split a 64-bit hash into bucket index and lower bits.
///
/// # Arguments
///
/// * `hash` - The full 64-bit hash
/// * `num_buckets` - Number of buckets in the index
/// * `hash_lower_bits` - Number of bits to store in entry
///
/// # Returns
///
/// Tuple of (bucket_idx, lower_hash)
pub fn split_hash(hash: u64, num_buckets: u32, hash_lower_bits: u32) -> (u32, u64) {
    let lower_mask = (1u64 << hash_lower_bits) - 1;
    let lower_hash = hash & lower_mask;
    let bucket_idx = ((hash >> hash_lower_bits) as u32) % num_buckets;
    (bucket_idx, lower_hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_header() -> S3IndexHeader {
        S3IndexHeader {
            version: 1,
            num_buckets: 1024,
            num_entries: 0,
            num_files: 10,
            hash_upper_bits: 32,
            hash_lower_bits: 32,
            seg_id_bits: 4,
            row_id_bits: 24,
            entry_size: 8, // (32 + 4 + 24) / 8 = 7.5 → 8
            bucket_dir_offset: 0,
            bucket_dir_size: 0,
            entry_block_offset: 0,
            entry_block_size: 0,
            file_list_offset: 0,
            file_list_size: 0,
            checksum: 0,
        }
    }

    #[test]
    fn test_entry_roundtrip() {
        let header = make_test_header();

        let entry = S3IndexEntry {
            lower_hash: 0x12345678,
            file_idx: 5,
            row_idx: 12345,
        };

        let encoded = entry.encode(&header);
        assert_eq!(encoded.len(), header.entry_size as usize);

        let decoded = S3IndexEntry::decode(&encoded, &header).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn test_entry_max_values() {
        let header = make_test_header();

        // Max values for each field
        let entry = S3IndexEntry {
            lower_hash: (1u64 << 32) - 1, // Max 32-bit value
            file_idx: (1u32 << 4) - 1,    // Max 4-bit value (15)
            row_idx: (1u64 << 24) - 1,    // Max 24-bit value
        };

        let encoded = entry.encode(&header);
        let decoded = S3IndexEntry::decode(&encoded, &header).unwrap();
        assert_eq!(entry, decoded);
    }

    #[test]
    fn test_multiple_entries_roundtrip() {
        let header = make_test_header();

        let entries = vec![
            S3IndexEntry::new(0x11111111, 0, 100),
            S3IndexEntry::new(0x22222222, 1, 200),
            S3IndexEntry::new(0x33333333, 2, 300),
            S3IndexEntry::new(0x44444444, 3, 400),
        ];

        let encoded = S3IndexEntry::encode_all(&entries, &header);
        assert_eq!(encoded.len(), entries.len() * header.entry_size as usize);

        let decoded = S3IndexEntry::decode_all(&encoded, &header).unwrap();
        assert_eq!(entries, decoded);
    }

    #[test]
    fn test_split_and_reconstruct_hash() {
        let hash: u64 = 0x1234567890ABCDEF;
        let num_buckets = 1024;
        let hash_lower_bits = 32;

        let (bucket_idx, lower_hash) = split_hash(hash, num_buckets, hash_lower_bits);

        // Bucket should be (hash >> 32) % 1024
        assert_eq!(bucket_idx, (0x12345678u32) % 1024);
        assert_eq!(lower_hash, 0x90ABCDEF);

        // Note: reconstruct_hash doesn't account for modulo, so it won't perfectly
        // reconstruct the original hash if num_buckets < 2^hash_upper_bits
        let reconstructed = reconstruct_hash(bucket_idx, lower_hash, hash_lower_bits);
        // The lower bits should match
        assert_eq!(reconstructed & 0xFFFFFFFF, hash & 0xFFFFFFFF);
    }

    #[test]
    fn test_small_entry_size() {
        // Test with smaller bit widths
        let header = S3IndexHeader {
            version: 1,
            num_buckets: 256,
            num_entries: 0,
            num_files: 4,
            hash_upper_bits: 48,
            hash_lower_bits: 16, // Only 16 bits of hash
            seg_id_bits: 2,      // 4 files max
            row_id_bits: 14,     // 16K rows max
            entry_size: 4,       // (16 + 2 + 14) / 8 = 4 bytes
            bucket_dir_offset: 0,
            bucket_dir_size: 0,
            entry_block_offset: 0,
            entry_block_size: 0,
            file_list_offset: 0,
            file_list_size: 0,
            checksum: 0,
        };

        let entry = S3IndexEntry {
            lower_hash: 0xABCD,
            file_idx: 3,
            row_idx: 8000,
        };

        let encoded = entry.encode(&header);
        assert_eq!(encoded.len(), 4);

        let decoded = S3IndexEntry::decode(&encoded, &header).unwrap();
        assert_eq!(entry, decoded);
    }
}
