//! # S3-Based Global Index
//!
//! This module implements a cloud-native index optimized for S3 storage,
//! providing fast primary key lookups for Iceberg tables without local disk.
//!
//! ## Overview
//!
//! Unlike [`GlobalIndex`](super::persisted_bucket_hash_map::GlobalIndex) which uses
//! memory-mapped local files, `S3GlobalIndex` stores index data directly on S3 and
//! uses range requests for efficient lookups.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      S3GlobalIndex                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  In-Memory (always cached):                                      │
//! │  ├─ Header (fixed 256 bytes)                                    │
//! │  ├─ Bucket Directory (bucket → byte offset)                     │
//! │  └─ File List (data files covered)                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  In-Memory (LRU cache, configurable size):                       │
//! │  └─ Entry Cache (bucket_idx → entries)                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  On S3:                                                          │
//! │  └─ Index File (header + file list + bucket dir + entries)      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## File Format
//!
//! The index file is optimized for S3 range requests:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │ S3 Index File Layout                                            │
//! ├──────────────────────────────────────────────────────────────────┤
//! │ HEADER (256 bytes)          ← Always fetched first              │
//! │ FILE LIST (variable)        ← Fetched with header               │
//! │ BUCKET DIRECTORY (fixed 8B/bucket) ← Fetched once              │
//! │ ENTRIES (fixed size each)   ← Fetched via range requests       │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Lookup Flow
//!
//! ```text
//! 1. Hash key → bucket index
//! 2. Check LRU cache for bucket entries
//! 3. Cache miss? → S3 range request for bucket entries
//! 4. Filter entries by lower hash bits
//! 5. Return matching record locations
//! ```
//!
//! ## Use Case
//!
//! This index is designed for:
//! - **Fast CDC ingestion**: Looking up existing records during Postgres → Iceberg sync
//! - **Serverless environments**: No local disk required
//! - **Large datasets**: Indices can be GBs, only relevant parts are fetched
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! // Create S3 client
//! let s3_client = MyS3Client::new("my-bucket", "us-east-1");
//!
//! // Build index from hash entries
//! let builder = S3GlobalIndexBuilder::new(
//!     Arc::new(s3_client),
//!     S3IndexConfig {
//!         cache_size_bytes: 100 * 1024 * 1024, // 100 MB cache
//!         num_buckets: 65536,
//!         s3_bucket: "my-bucket".into(),
//!         s3_prefix: "indices/table_123".into(),
//!     },
//! );
//!
//! let index = builder
//!     .set_files(vec!["data1.parquet".into(), "data2.parquet".into()])
//!     .build_from_flush(hash_entries, index_id)
//!     .await?;
//!
//! // Lookup a key
//! let locations = index.find(12345).await?;
//!
//! // Batch lookup (more efficient)
//! let results = index.find_batch(&[100, 200, 300]).await?;
//! ```
//!
//! ## Merging
//!
//! Multiple indices can be merged into one:
//!
//! ```rust,ignore
//! let merged = builder
//!     .build_from_merge(&[&index1, &index2, &index3], new_index_id)
//!     .await?;
//! ```

mod builder;
mod cache;
mod client;
mod config;
mod entry;
mod error;
mod format;
mod index;
mod reader;

pub use builder::S3GlobalIndexBuilder;
pub use cache::{S3IndexCache, S3IndexCacheConfig};
pub use client::S3Client;
pub use config::S3IndexConfig;
pub use entry::S3IndexEntry;
pub use error::{S3IndexError, S3IndexResult};
pub use format::{BucketInfo, S3IndexHeader, HEADER_SIZE, MAGIC_NUMBER};
pub use index::S3GlobalIndex;

#[cfg(test)]
pub(crate) mod mock_client;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integration_tests;
