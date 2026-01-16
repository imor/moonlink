//! Mock S3 client for testing.
//!
//! This module provides an in-memory S3 client implementation for unit tests.

use async_trait::async_trait;
use bytes::Bytes;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::RwLock;

use super::client::S3Client;
use super::error::{S3IndexError, S3IndexResult};

/// In-memory mock S3 client for testing.
///
/// Stores objects in a `HashMap` and supports all `S3Client` operations.
///
/// # Example
///
/// ```rust,ignore
/// let mock = MockS3Client::new();
///
/// // Upload an object
/// mock.put_object("test/file.bin", Bytes::from("hello")).await?;
///
/// // Read it back
/// let data = mock.get_object("test/file.bin").await?;
/// assert_eq!(&data[..], b"hello");
///
/// // Range request
/// let partial = mock.get_range("test/file.bin", 0..2).await?;
/// assert_eq!(&partial[..], b"he");
/// ```
pub struct MockS3Client {
    /// Storage for objects: key -> data
    objects: RwLock<HashMap<String, Bytes>>,
}

impl MockS3Client {
    /// Create a new empty mock client.
    pub fn new() -> Self {
        Self {
            objects: RwLock::new(HashMap::new()),
        }
    }

    /// Create a mock client with pre-populated objects.
    pub fn with_objects(objects: HashMap<String, Bytes>) -> Self {
        Self {
            objects: RwLock::new(objects),
        }
    }

    /// Get all object keys (for debugging/testing).
    pub fn keys(&self) -> Vec<String> {
        self.objects.read().unwrap().keys().cloned().collect()
    }

    /// Check if an object exists (synchronous, for tests).
    pub fn contains(&self, key: &str) -> bool {
        self.objects.read().unwrap().contains_key(key)
    }

    /// Get object data directly (synchronous, for tests).
    pub fn get_data(&self, key: &str) -> Option<Bytes> {
        self.objects.read().unwrap().get(key).cloned()
    }

    /// Clear all objects.
    pub fn clear(&self) {
        self.objects.write().unwrap().clear();
    }
}

impl Default for MockS3Client {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl S3Client for MockS3Client {
    async fn get_range(&self, key: &str, range: Range<u64>) -> S3IndexResult<Bytes> {
        let objects = self.objects.read().unwrap();
        let data = objects
            .get(key)
            .ok_or_else(|| S3IndexError::ObjectNotFound {
                key: key.to_string(),
            })?;

        let start = range.start as usize;
        let end = range.end as usize;

        if start > data.len() || end > data.len() || start > end {
            return Err(S3IndexError::InvalidRange {
                key: key.to_string(),
                start: range.start,
                end: range.end,
                object_size: data.len() as u64,
            });
        }

        Ok(data.slice(start..end))
    }

    async fn put_object(&self, key: &str, data: Bytes) -> S3IndexResult<()> {
        let mut objects = self.objects.write().unwrap();
        objects.insert(key.to_string(), data);
        Ok(())
    }

    async fn delete_object(&self, key: &str) -> S3IndexResult<()> {
        let mut objects = self.objects.write().unwrap();
        objects.remove(key);
        Ok(())
    }

    async fn head_object(&self, key: &str) -> S3IndexResult<Option<u64>> {
        let objects = self.objects.read().unwrap();
        Ok(objects.get(key).map(|data| data.len() as u64))
    }
}

/// Builder for creating mock S3 clients with test data.
pub struct MockS3ClientBuilder {
    objects: HashMap<String, Bytes>,
}

impl MockS3ClientBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }

    /// Add an object.
    pub fn with_object(mut self, key: impl Into<String>, data: impl Into<Bytes>) -> Self {
        self.objects.insert(key.into(), data.into());
        self
    }

    /// Build the mock client.
    pub fn build(self) -> MockS3Client {
        MockS3Client::with_objects(self.objects)
    }
}

impl Default for MockS3ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_put_get() {
        let client = MockS3Client::new();

        client
            .put_object("test/file.txt", Bytes::from("hello world"))
            .await
            .unwrap();

        let data = client.get_object("test/file.txt").await.unwrap();
        assert_eq!(&data[..], b"hello world");
    }

    #[tokio::test]
    async fn test_mock_range_request() {
        let client = MockS3Client::new();

        client
            .put_object("test/file.txt", Bytes::from("hello world"))
            .await
            .unwrap();

        // Read "hello"
        let data = client.get_range("test/file.txt", 0..5).await.unwrap();
        assert_eq!(&data[..], b"hello");

        // Read "world"
        let data = client.get_range("test/file.txt", 6..11).await.unwrap();
        assert_eq!(&data[..], b"world");
    }

    #[tokio::test]
    async fn test_mock_not_found() {
        let client = MockS3Client::new();

        let result = client.get_object("nonexistent").await;
        assert!(matches!(result, Err(S3IndexError::ObjectNotFound { .. })));
    }

    #[tokio::test]
    async fn test_mock_invalid_range() {
        let client = MockS3Client::new();

        client
            .put_object("test/file.txt", Bytes::from("hello"))
            .await
            .unwrap();

        let result = client.get_range("test/file.txt", 0..100).await;
        assert!(matches!(result, Err(S3IndexError::InvalidRange { .. })));
    }

    #[tokio::test]
    async fn test_mock_delete() {
        let client = MockS3Client::new();

        client
            .put_object("test/file.txt", Bytes::from("hello"))
            .await
            .unwrap();
        assert!(client.contains("test/file.txt"));

        client.delete_object("test/file.txt").await.unwrap();
        assert!(!client.contains("test/file.txt"));

        // Delete non-existent should not error
        client.delete_object("test/file.txt").await.unwrap();
    }

    #[tokio::test]
    async fn test_mock_head() {
        let client = MockS3Client::new();

        let size = client.head_object("test/file.txt").await.unwrap();
        assert_eq!(size, None);

        client
            .put_object("test/file.txt", Bytes::from("hello"))
            .await
            .unwrap();

        let size = client.head_object("test/file.txt").await.unwrap();
        assert_eq!(size, Some(5));
    }

    #[tokio::test]
    async fn test_mock_builder() {
        let client = MockS3ClientBuilder::new()
            .with_object("file1.txt", "content1")
            .with_object("file2.txt", "content2")
            .build();

        assert!(client.contains("file1.txt"));
        assert!(client.contains("file2.txt"));

        let data = client.get_object("file1.txt").await.unwrap();
        assert_eq!(&data[..], b"content1");
    }
}
