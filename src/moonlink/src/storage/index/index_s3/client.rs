//! S3 Client trait for abstracting S3 operations.
//!
//! This module defines the [`S3Client`] trait which abstracts S3 operations
//! for testability and flexibility in choosing S3 client implementations.

use async_trait::async_trait;
use bytes::Bytes;
use std::ops::Range;

use super::error::S3IndexResult;

/// Abstraction over S3 operations for index storage.
///
/// This trait allows for different S3 client implementations:
/// - Production: aws-sdk-rust, opendal, etc.
/// - Testing: Mock client, LocalStack client
///
/// # Example Implementation
///
/// ```rust,ignore
/// use aws_sdk_s3::Client;
///
/// struct AwsS3Client {
///     client: Client,
///     bucket: String,
/// }
///
/// #[async_trait]
/// impl S3Client for AwsS3Client {
///     async fn get_range(&self, key: &str, range: Range<u64>) -> S3IndexResult<Bytes> {
///         let resp = self.client
///             .get_object()
///             .bucket(&self.bucket)
///             .key(key)
///             .range(format!("bytes={}-{}", range.start, range.end - 1))
///             .send()
///             .await?;
///         Ok(resp.body.collect().await?.into_bytes())
///     }
///     // ... other methods
/// }
/// ```
#[async_trait]
pub trait S3Client: Send + Sync {
    /// Read a byte range from an S3 object.
    ///
    /// # Arguments
    ///
    /// * `key` - S3 object key (path within bucket)
    /// * `range` - Byte range to read (start inclusive, end exclusive)
    ///
    /// # Returns
    ///
    /// The requested bytes from the object.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Object doesn't exist
    /// - Range is invalid
    /// - Network/permission issues
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Read bytes 100-199 (100 bytes total)
    /// let data = client.get_range("indices/idx_001.bin", 100..200).await?;
    /// assert_eq!(data.len(), 100);
    /// ```
    async fn get_range(&self, key: &str, range: Range<u64>) -> S3IndexResult<Bytes>;

    /// Write an object to S3.
    ///
    /// # Arguments
    ///
    /// * `key` - S3 object key (path within bucket)
    /// * `data` - Content to write
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Permission denied
    /// - Network issues
    /// - Bucket doesn't exist
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let data = build_index_bytes(&entries);
    /// client.put_object("indices/idx_001.bin", data).await?;
    /// ```
    async fn put_object(&self, key: &str, data: Bytes) -> S3IndexResult<()>;

    /// Delete an object from S3.
    ///
    /// # Arguments
    ///
    /// * `key` - S3 object key to delete
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Permission denied
    /// - Network issues
    ///
    /// # Note
    ///
    /// This operation is idempotent - deleting a non-existent object is not an error.
    async fn delete_object(&self, key: &str) -> S3IndexResult<()>;

    /// Check if an object exists and get its size.
    ///
    /// # Arguments
    ///
    /// * `key` - S3 object key to check
    ///
    /// # Returns
    ///
    /// - `Some(size)` if object exists
    /// - `None` if object doesn't exist
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Permission denied
    /// - Network issues
    async fn head_object(&self, key: &str) -> S3IndexResult<Option<u64>>;

    /// Get the full content of an object.
    ///
    /// # Arguments
    ///
    /// * `key` - S3 object key
    ///
    /// # Returns
    ///
    /// The complete object content.
    ///
    /// # Note
    ///
    /// For large objects, prefer using `get_range` to fetch only needed parts.
    async fn get_object(&self, key: &str) -> S3IndexResult<Bytes> {
        // Default implementation: get size then fetch full range
        if let Some(size) = self.head_object(key).await? {
            self.get_range(key, 0..size).await
        } else {
            Err(super::error::S3IndexError::ObjectNotFound {
                key: key.to_string(),
            })
        }
    }
}
