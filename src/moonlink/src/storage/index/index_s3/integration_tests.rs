//! Integration tests for S3 index with LocalStack/MinIO.
//!
//! These tests require a running LocalStack or MinIO instance.
//!
//! To run with LocalStack:
//! ```bash
//! # Start LocalStack
//! docker run -d -p 4566:4566 localstack/localstack
//!
//! # Set environment variables
//! export S3_INDEX_TEST_ENDPOINT=http://localhost:4566
//! export S3_INDEX_TEST_BUCKET=test-bucket
//! export AWS_ACCESS_KEY_ID=test
//! export AWS_SECRET_ACCESS_KEY=test
//! export AWS_DEFAULT_REGION=us-east-1
//!
//! # Run tests
//! cargo test --features storage-s3 s3_integration_tests -- --ignored
//! ```
//!
//! To run with MinIO:
//! ```bash
//! # Start MinIO
//! docker run -d -p 9000:9000 -p 9001:9001 \
//!   -e MINIO_ROOT_USER=minioadmin \
//!   -e MINIO_ROOT_PASSWORD=minioadmin \
//!   minio/minio server /data --console-address ":9001"
//!
//! # Set environment variables
//! export S3_INDEX_TEST_ENDPOINT=http://localhost:9000
//! export S3_INDEX_TEST_BUCKET=test-bucket
//! export AWS_ACCESS_KEY_ID=minioadmin
//! export AWS_SECRET_ACCESS_KEY=minioadmin
//! export AWS_DEFAULT_REGION=us-east-1
//!
//! # Create bucket first (via MinIO console or mc)
//! # Then run tests
//! cargo test --features storage-s3 s3_integration_tests -- --ignored
//! ```

// This module contains integration test scaffolding.
// Actual integration tests are placed in the tests/ directory or can be
// enabled with feature flags.

#[cfg(test)]
mod localstack_client {
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::ops::Range;
    use std::env;

    use crate::storage::index::index_s3::client::S3Client;
    use crate::storage::index::index_s3::error::{S3IndexError, S3IndexResult};

    /// Real S3 client implementation using reqwest for LocalStack/MinIO testing.
    ///
    /// This is a minimal implementation for integration testing.
    /// In production, you'd use aws-sdk-s3 or opendal.
    pub struct LocalStackS3Client {
        endpoint: String,
        bucket: String,
        client: reqwest::Client,
    }

    impl LocalStackS3Client {
        /// Create a new client from environment variables.
        pub fn from_env() -> Option<Self> {
            let endpoint = env::var("S3_INDEX_TEST_ENDPOINT").ok()?;
            let bucket = env::var("S3_INDEX_TEST_BUCKET").ok()?;

            Some(Self {
                endpoint,
                bucket,
                client: reqwest::Client::new(),
            })
        }

        /// Create a new client with explicit configuration.
        pub fn new(endpoint: impl Into<String>, bucket: impl Into<String>) -> Self {
            Self {
                endpoint: endpoint.into(),
                bucket: bucket.into(),
                client: reqwest::Client::new(),
            }
        }

        fn object_url(&self, key: &str) -> String {
            format!("{}/{}/{}", self.endpoint, self.bucket, key)
        }
    }

    #[async_trait]
    impl S3Client for LocalStackS3Client {
        async fn get_range(&self, key: &str, range: Range<u64>) -> S3IndexResult<Bytes> {
            let url = self.object_url(key);
            let range_header = format!("bytes={}-{}", range.start, range.end - 1);

            let response = self.client
                .get(&url)
                .header("Range", range_header)
                .send()
                .await
                .map_err(|e| S3IndexError::s3_operation("get_range", key, e.to_string()))?;

            if response.status() == reqwest::StatusCode::NOT_FOUND {
                return Err(S3IndexError::ObjectNotFound { key: key.to_string() });
            }

            if !response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
                return Err(S3IndexError::s3_operation(
                    "get_range",
                    key,
                    format!("HTTP {}", response.status()),
                ));
            }

            let bytes = response
                .bytes()
                .await
                .map_err(|e| S3IndexError::s3_operation("get_range", key, e.to_string()))?;

            Ok(bytes)
        }

        async fn put_object(&self, key: &str, data: Bytes) -> S3IndexResult<()> {
            let url = self.object_url(key);

            let response = self.client
                .put(&url)
                .body(data)
                .send()
                .await
                .map_err(|e| S3IndexError::s3_operation("put_object", key, e.to_string()))?;

            if !response.status().is_success() {
                return Err(S3IndexError::s3_operation(
                    "put_object",
                    key,
                    format!("HTTP {}", response.status()),
                ));
            }

            Ok(())
        }

        async fn delete_object(&self, key: &str) -> S3IndexResult<()> {
            let url = self.object_url(key);

            let response = self.client
                .delete(&url)
                .send()
                .await
                .map_err(|e| S3IndexError::s3_operation("delete_object", key, e.to_string()))?;

            // 204 No Content is success for DELETE
            if !response.status().is_success() && response.status() != reqwest::StatusCode::NOT_FOUND {
                return Err(S3IndexError::s3_operation(
                    "delete_object",
                    key,
                    format!("HTTP {}", response.status()),
                ));
            }

            Ok(())
        }

        async fn head_object(&self, key: &str) -> S3IndexResult<Option<u64>> {
            let url = self.object_url(key);

            let response = self.client
                .head(&url)
                .send()
                .await
                .map_err(|e| S3IndexError::s3_operation("head_object", key, e.to_string()))?;

            if response.status() == reqwest::StatusCode::NOT_FOUND {
                return Ok(None);
            }

            if !response.status().is_success() {
                return Err(S3IndexError::s3_operation(
                    "head_object",
                    key,
                    format!("HTTP {}", response.status()),
                ));
            }

            let size = response
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok());

            Ok(size)
        }
    }
}

/// Integration tests that run against LocalStack/MinIO.
///
/// These tests are ignored by default and require environment setup.
#[cfg(test)]
mod integration_tests {
    use std::sync::Arc;

    use super::localstack_client::LocalStackS3Client;
    use crate::storage::index::index_s3::builder::S3GlobalIndexBuilder;
    use crate::storage::index::index_s3::config::S3IndexConfig;
    use crate::storage::index::index_s3::reader::splitmix64;

    fn get_test_client() -> Option<Arc<LocalStackS3Client>> {
        LocalStackS3Client::from_env().map(Arc::new)
    }

    fn test_config() -> S3IndexConfig {
        let bucket = std::env::var("S3_INDEX_TEST_BUCKET").unwrap_or_else(|_| "test-bucket".into());
        S3IndexConfig::new(bucket, format!("test/s3_index/{}", uuid::Uuid::new_v4()))
            .with_cache_size(10 * 1024 * 1024)
            .with_num_buckets(64)
    }

    #[tokio::test]
    #[ignore = "Requires LocalStack/MinIO - set S3_INDEX_TEST_ENDPOINT and S3_INDEX_TEST_BUCKET"]
    async fn test_localstack_build_and_lookup() {
        let Some(client) = get_test_client() else {
            eprintln!("Skipping test: S3 test environment not configured");
            return;
        };

        let config = test_config();

        let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
            .set_files(vec!["data/file1.parquet".to_string()]);

        let entries: Vec<_> = (0..100)
            .map(|i| (splitmix64(i as u64), 0, i as u64))
            .collect();

        let index = builder.build_from_flush(entries, 1).await.unwrap();

        // Test lookups
        for key in [0, 50, 99] {
            let locations = index.find(key).await.unwrap();
            assert_eq!(locations.len(), 1, "Key {} should have 1 location", key);
            assert_eq!(locations[0].row_idx, key);
        }

        // Test non-existent key
        let locations = index.find(1000).await.unwrap();
        assert!(locations.is_empty());

        // Cleanup
        index.delete().await.unwrap();
    }

    #[tokio::test]
    #[ignore = "Requires LocalStack/MinIO - set S3_INDEX_TEST_ENDPOINT and S3_INDEX_TEST_BUCKET"]
    async fn test_localstack_batch_lookup() {
        let Some(client) = get_test_client() else {
            return;
        };

        let config = test_config();

        let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
            .set_files(vec!["data.parquet".to_string()]);

        let entries: Vec<_> = (0..1000)
            .map(|i| (splitmix64(i as u64), 0, i as u64))
            .collect();

        let index = builder.build_from_flush(entries, 1).await.unwrap();

        // Batch lookup
        let keys: Vec<u64> = (0..100).step_by(7).collect();
        let results = index.find_batch(&keys).await.unwrap();

        assert_eq!(results.len(), keys.len());
        for (key, locations) in results {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].row_idx, key);
        }

        // Cleanup
        index.delete().await.unwrap();
    }

    #[tokio::test]
    #[ignore = "Requires LocalStack/MinIO - set S3_INDEX_TEST_ENDPOINT and S3_INDEX_TEST_BUCKET"]
    async fn test_localstack_merge_indices() {
        let Some(client) = get_test_client() else {
            return;
        };

        let config = test_config();

        // Create first index
        let builder1 = S3GlobalIndexBuilder::new(client.clone(), config.clone())
            .set_files(vec!["file1.parquet".to_string()]);
        let entries1: Vec<_> = (0..50)
            .map(|i| (splitmix64(i as u64), 0, i as u64))
            .collect();
        let index1 = builder1.build_from_flush(entries1, 1).await.unwrap();

        // Create second index
        let builder2 = S3GlobalIndexBuilder::new(client.clone(), config.clone())
            .set_files(vec!["file2.parquet".to_string()]);
        let entries2: Vec<_> = (50..100)
            .map(|i| (splitmix64(i as u64), 0, (i - 50) as u64))
            .collect();
        let index2 = builder2.build_from_flush(entries2, 2).await.unwrap();

        // Merge indices
        let merge_builder = S3GlobalIndexBuilder::new(client.clone(), config.clone());
        let merged = merge_builder
            .build_from_merge(&[&index1, &index2], 3)
            .await
            .unwrap();

        // Verify merged index contains entries from both
        assert_eq!(merged.num_entries().await.unwrap(), 100);

        // Check entry from first index
        let loc = &merged.find(25).await.unwrap()[0];
        assert_eq!(loc.file_path, "file1.parquet");

        // Check entry from second index
        let loc = &merged.find(75).await.unwrap()[0];
        assert_eq!(loc.file_path, "file2.parquet");

        // Cleanup
        index1.delete().await.unwrap();
        index2.delete().await.unwrap();
        merged.delete().await.unwrap();
    }

    #[tokio::test]
    #[ignore = "Requires LocalStack/MinIO - set S3_INDEX_TEST_ENDPOINT and S3_INDEX_TEST_BUCKET"]
    async fn test_localstack_large_index() {
        let Some(client) = get_test_client() else {
            return;
        };

        let config = S3IndexConfig::new(
            std::env::var("S3_INDEX_TEST_BUCKET").unwrap_or_else(|_| "test-bucket".into()),
            format!("test/s3_index/{}", uuid::Uuid::new_v4()),
        )
        .with_cache_size(1024 * 1024) // 1 MB cache
        .with_num_buckets(1024);

        let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
            .set_files(vec!["large_data.parquet".to_string()]);

        // Create 100K entries
        let entries: Vec<_> = (0..100_000)
            .map(|i| (splitmix64(i as u64), 0, i as u64))
            .collect();

        let index = builder.build_from_flush(entries, 1).await.unwrap();

        assert_eq!(index.num_entries().await.unwrap(), 100_000);

        // Sample some lookups
        for key in [0, 10_000, 50_000, 99_999] {
            let locations = index.find(key).await.unwrap();
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].row_idx, key);
        }

        // Cleanup
        index.delete().await.unwrap();
    }
}
