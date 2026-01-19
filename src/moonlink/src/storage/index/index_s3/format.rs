//! Index file format structures.
//!
//! This module defines the binary format for S3 index files, optimized for
//! S3 range requests.

use bytes::{Buf, BufMut, Bytes, BytesMut};

use super::error::{S3IndexError, S3IndexResult};

/// Magic number for S3 index files: "MOONIDX1"
pub const MAGIC_NUMBER: [u8; 8] = *b"MOONIDX1";

/// Current index format version.
pub const CURRENT_VERSION: u32 = 1;

/// Fixed header size in bytes.
pub const HEADER_SIZE: u64 = 256;

/// Size of each bucket directory entry in bytes (offset: u64).
pub const BUCKET_DIR_ENTRY_SIZE: u64 = 8;

/// Index file header containing metadata and section offsets.
///
/// The header is always at the start of the file and has a fixed size
/// for predictable S3 range requests.
///
/// # Layout (256 bytes)
///
/// ```text
/// Offset  Size  Field
/// ------  ----  -----
/// 0       8     Magic number ("MOONIDX1")
/// 8       4     Version
/// 12      4     Number of buckets
/// 16      8     Number of entries
/// 24      4     Number of files
/// 28      4     Hash upper bits (for bucket selection)
/// 32      4     Hash lower bits (stored in entries)
/// 36      4     Segment ID bits
/// 40      4     Row ID bits
/// 44      4     Entry size (bytes)
/// 48      8     Bucket directory offset
/// 56      8     Bucket directory size
/// 64      8     Entry block offset
/// 72      8     Entry block size
/// 80      8     File list offset
/// 88      8     File list size
/// 96      8     Checksum (CRC64 of header bytes 0-95)
/// 104     152   Reserved (zero-filled)
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct S3IndexHeader {
    /// Format version.
    pub version: u32,

    /// Number of hash buckets.
    pub num_buckets: u32,

    /// Total number of entries in the index.
    pub num_entries: u64,

    /// Number of data files covered by this index.
    pub num_files: u32,

    /// Bits used for bucket selection (upper part of hash).
    pub hash_upper_bits: u32,

    /// Bits stored in each entry (lower part of hash).
    pub hash_lower_bits: u32,

    /// Bits for segment (file) index.
    pub file_id_bits: u32,

    /// Bits for row index within file.
    pub row_id_bits: u32,

    /// Size of each entry in bytes (derived from bit fields).
    pub entry_size: u32,

    /// Byte offset where bucket directory starts.
    pub bucket_dir_offset: u64,

    /// Size of bucket directory in bytes.
    pub bucket_dir_size: u64,

    /// Byte offset where entries block starts.
    pub entry_block_offset: u64,

    /// Size of entries block in bytes.
    pub entry_block_size: u64,

    /// Byte offset where file list starts.
    pub file_list_offset: u64,

    /// Size of file list in bytes.
    pub file_list_size: u64,

    /// CRC64 checksum of header (excluding checksum field itself).
    pub checksum: u64,
}

impl S3IndexHeader {
    /// Encode the header to bytes.
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(HEADER_SIZE as usize);

        // Magic number
        buf.put_slice(&MAGIC_NUMBER);

        // Version and counts
        buf.put_u32(self.version);
        buf.put_u32(self.num_buckets);
        buf.put_u64(self.num_entries);
        buf.put_u32(self.num_files);

        // Bit widths
        buf.put_u32(self.hash_upper_bits);
        buf.put_u32(self.hash_lower_bits);
        buf.put_u32(self.file_id_bits);
        buf.put_u32(self.row_id_bits);
        buf.put_u32(self.entry_size);

        // Section offsets
        buf.put_u64(self.bucket_dir_offset);
        buf.put_u64(self.bucket_dir_size);
        buf.put_u64(self.entry_block_offset);
        buf.put_u64(self.entry_block_size);
        buf.put_u64(self.file_list_offset);
        buf.put_u64(self.file_list_size);

        // Calculate checksum over bytes 0-95
        let checksum = crc64_checksum(&buf[..96]);
        buf.put_u64(checksum);

        // Reserved space (fill to 256 bytes)
        let reserved_size = HEADER_SIZE as usize - buf.len();
        buf.put_bytes(0, reserved_size);

        buf.freeze()
    }

    /// Decode header from bytes.
    pub fn decode(data: &[u8]) -> S3IndexResult<Self> {
        if data.len() < HEADER_SIZE as usize {
            return Err(S3IndexError::invalid_format(format!(
                "Header too small: {} bytes, expected {}",
                data.len(),
                HEADER_SIZE
            )));
        }

        let mut buf = &data[..];

        // Magic number
        let mut magic = [0u8; 8];
        buf.copy_to_slice(&mut magic);
        if magic != MAGIC_NUMBER {
            return Err(S3IndexError::InvalidMagicNumber {
                expected: MAGIC_NUMBER,
                found: magic,
            });
        }

        // Version
        let version = buf.get_u32();
        if version > CURRENT_VERSION {
            return Err(S3IndexError::UnsupportedVersion {
                version,
                max_supported: CURRENT_VERSION,
            });
        }

        // Counts
        let num_buckets = buf.get_u32();
        let num_entries = buf.get_u64();
        let num_files = buf.get_u32();

        // Bit widths
        let hash_upper_bits = buf.get_u32();
        let hash_lower_bits = buf.get_u32();
        let file_id_bits = buf.get_u32();
        let row_id_bits = buf.get_u32();
        let entry_size = buf.get_u32();

        // Section offsets
        let bucket_dir_offset = buf.get_u64();
        let bucket_dir_size = buf.get_u64();
        let entry_block_offset = buf.get_u64();
        let entry_block_size = buf.get_u64();
        let file_list_offset = buf.get_u64();
        let file_list_size = buf.get_u64();

        // Checksum
        let stored_checksum = buf.get_u64();
        let computed_checksum = crc64_checksum(&data[..96]);
        if stored_checksum != computed_checksum {
            return Err(S3IndexError::ChecksumMismatch {
                expected: stored_checksum,
                actual: computed_checksum,
            });
        }

        Ok(Self {
            version,
            num_buckets,
            num_entries,
            num_files,
            hash_upper_bits,
            hash_lower_bits,
            file_id_bits,
            row_id_bits,
            entry_size,
            bucket_dir_offset,
            bucket_dir_size,
            entry_block_offset,
            entry_block_size,
            file_list_offset,
            file_list_size,
            checksum: stored_checksum,
        })
    }

    /// Calculate entry size from bit widths.
    ///
    /// Each entry contains: lower_hash + file_id + row_id
    /// Size is rounded up to the nearest byte.
    pub fn calculate_entry_size(hash_lower_bits: u32, file_id_bits: u32, row_id_bits: u32) -> u32 {
        let total_bits = hash_lower_bits + file_id_bits + row_id_bits;
        (total_bits + 7) / 8 // Round up to bytes
    }
}

/// Bucket directory entry - maps bucket index to entry range.
///
/// Each bucket stores:
/// - The byte offset of its first entry (relative to entry block start)
/// - The count is implicitly derived from the next bucket's offset
///
/// For simplicity, we store the absolute entry offset for each bucket.
#[derive(Clone, Debug, PartialEq)]
pub struct BucketInfo {
    /// Byte offset from start of entry block where this bucket's entries begin.
    pub entry_offset: u64,
}

impl BucketInfo {
    /// Encode a bucket directory to bytes.
    ///
    /// Format: For each bucket, store the starting entry offset (8 bytes each).
    /// An extra entry at the end stores the total size for calculating last bucket's count.
    pub fn encode_directory(buckets: &[BucketInfo]) -> Bytes {
        let mut buf = BytesMut::with_capacity((buckets.len() + 1) * BUCKET_DIR_ENTRY_SIZE as usize);

        for bucket in buckets {
            buf.put_u64(bucket.entry_offset);
        }

        buf.freeze()
    }

    /// Decode bucket directory from bytes.
    pub fn decode_directory(data: &[u8], num_buckets: u32) -> S3IndexResult<Vec<BucketInfo>> {
        let expected_size = (num_buckets as usize + 1) * BUCKET_DIR_ENTRY_SIZE as usize;
        if data.len() < expected_size {
            return Err(S3IndexError::invalid_format(format!(
                "Bucket directory too small: {} bytes, expected {}",
                data.len(),
                expected_size
            )));
        }

        let mut buf = &data[..];
        let mut buckets = Vec::with_capacity(num_buckets as usize);

        for _ in 0..=num_buckets {
            buckets.push(BucketInfo {
                entry_offset: buf.get_u64(),
            });
        }

        Ok(buckets)
    }
}

/// Compute CRC64 checksum (using crc32fast as base, doubled for 64-bit).
fn crc64_checksum(data: &[u8]) -> u64 {
    let crc1 = crc32fast::hash(data) as u64;
    let crc2 = crc32fast::hash(&data.iter().rev().copied().collect::<Vec<_>>()) as u64;
    (crc1 << 32) | crc2
}

/// Encode file list as JSON for simplicity and forward compatibility.
pub fn encode_file_list(files: &[String]) -> S3IndexResult<Bytes> {
    let json = serde_json::to_vec(files)?;
    Ok(Bytes::from(json))
}

/// Decode file list from JSON.
pub fn decode_file_list(data: &[u8]) -> S3IndexResult<Vec<String>> {
    let files: Vec<String> = serde_json::from_slice(data)?;
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = S3IndexHeader {
            version: CURRENT_VERSION,
            num_buckets: 1024,
            num_entries: 1_000_000,
            num_files: 10,
            hash_upper_bits: 32,
            hash_lower_bits: 32,
            file_id_bits: 4,
            row_id_bits: 24,
            entry_size: 8,
            bucket_dir_offset: 256,
            bucket_dir_size: 8200,
            entry_block_offset: 8456,
            entry_block_size: 8_000_000,
            file_list_offset: 8_008_456,
            file_list_size: 500,
            checksum: 0, // Will be computed during encode
        };

        let encoded = header.encode();
        assert_eq!(encoded.len(), HEADER_SIZE as usize);

        let decoded = S3IndexHeader::decode(&encoded).unwrap();
        // Checksum is computed during encode, so we need to compare with that
        assert_eq!(decoded.version, header.version);
        assert_eq!(decoded.num_buckets, header.num_buckets);
        assert_eq!(decoded.num_entries, header.num_entries);
        assert_eq!(decoded.num_files, header.num_files);
        assert_eq!(decoded.hash_upper_bits, header.hash_upper_bits);
        assert_eq!(decoded.hash_lower_bits, header.hash_lower_bits);
        assert_eq!(decoded.file_id_bits, header.file_id_bits);
        assert_eq!(decoded.row_id_bits, header.row_id_bits);
        assert_eq!(decoded.entry_size, header.entry_size);
    }

    #[test]
    fn test_invalid_magic_number() {
        let mut data = vec![0u8; HEADER_SIZE as usize];
        data[..8].copy_from_slice(b"BADMAGIC");

        let result = S3IndexHeader::decode(&data);
        assert!(matches!(result, Err(S3IndexError::InvalidMagicNumber { .. })));
    }

    #[test]
    fn test_bucket_directory_roundtrip() {
        let buckets = vec![
            BucketInfo { entry_offset: 0 },
            BucketInfo { entry_offset: 100 },
            BucketInfo { entry_offset: 250 },
            BucketInfo { entry_offset: 250 }, // Empty bucket (same offset as next)
            BucketInfo { entry_offset: 400 }, // Sentinel
        ];

        let encoded = BucketInfo::encode_directory(&buckets);
        let decoded = BucketInfo::decode_directory(&encoded, 4).unwrap();

        assert_eq!(buckets, decoded);
    }

    #[test]
    fn test_file_list_roundtrip() {
        let files = vec![
            "s3://bucket/data/file1.parquet".to_string(),
            "s3://bucket/data/file2.parquet".to_string(),
            "s3://bucket/data/file3.parquet".to_string(),
        ];

        let encoded = encode_file_list(&files).unwrap();
        let decoded = decode_file_list(&encoded).unwrap();

        assert_eq!(files, decoded);
    }

    #[test]
    fn test_calculate_entry_size() {
        // 32 + 4 + 24 = 60 bits = 8 bytes
        assert_eq!(S3IndexHeader::calculate_entry_size(32, 4, 24), 8);

        // 32 + 8 + 32 = 72 bits = 9 bytes
        assert_eq!(S3IndexHeader::calculate_entry_size(32, 8, 32), 9);

        // 16 + 2 + 12 = 30 bits = 4 bytes
        assert_eq!(S3IndexHeader::calculate_entry_size(16, 2, 12), 4);
    }
}
