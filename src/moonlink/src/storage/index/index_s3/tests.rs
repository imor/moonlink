//! Unit tests for S3 index module.

use std::sync::Arc;

use super::builder::S3GlobalIndexBuilder;
use super::config::S3IndexConfig;
use super::mock_client::MockS3Client;
use super::reader::splitmix64;

/// Create a test configuration.
fn test_config() -> S3IndexConfig {
    S3IndexConfig::new("test-bucket", "test/indices")
        .with_cache_size(10 * 1024 * 1024) // 10 MB
        .with_num_buckets(64) // Small for tests
}

#[tokio::test]
async fn test_build_and_lookup_single_key() {
    let client = Arc::new(MockS3Client::new());
    let config = test_config();

    // Build index with a few entries
    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["file1.parquet".to_string()]);

    // Entries: (hash, file_idx, row_idx)
    let entries = vec![
        (splitmix64(100), 0, 0),
        (splitmix64(200), 0, 1),
        (splitmix64(300), 0, 2),
    ];

    let index = builder.build_from_flush(entries, 1).await.unwrap();

    // Verify index was uploaded
    assert!(client.contains(&config.index_key(1)));

    // Lookup existing key
    let locations = index.find(100).await.unwrap();
    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0].file_path, "file1.parquet");
    assert_eq!(locations[0].row_idx, 0);

    // Lookup another key
    let locations = index.find(300).await.unwrap();
    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0].row_idx, 2);

    // Lookup non-existent key
    let locations = index.find(999).await.unwrap();
    assert!(locations.is_empty());
}

#[tokio::test]
async fn test_build_and_lookup_multiple_files() {
    let client = Arc::new(MockS3Client::new());
    let config = test_config();

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone()).set_files(vec![
        "file1.parquet".to_string(),
        "file2.parquet".to_string(),
        "file3.parquet".to_string(),
    ]);

    let entries = vec![
        (splitmix64(100), 0, 0), // file1, row 0
        (splitmix64(200), 1, 5), // file2, row 5
        (splitmix64(300), 2, 10), // file3, row 10
    ];

    let index = builder.build_from_flush(entries, 1).await.unwrap();

    // Verify lookups return correct files
    let loc = &index.find(100).await.unwrap()[0];
    assert_eq!(loc.file_path, "file1.parquet");

    let loc = &index.find(200).await.unwrap()[0];
    assert_eq!(loc.file_path, "file2.parquet");
    assert_eq!(loc.row_idx, 5);

    let loc = &index.find(300).await.unwrap()[0];
    assert_eq!(loc.file_path, "file3.parquet");
    assert_eq!(loc.row_idx, 10);
}

#[tokio::test]
async fn test_batch_lookup() {
    let client = Arc::new(MockS3Client::new());
    let config = test_config();

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["data.parquet".to_string()]);

    let entries: Vec<_> = (0..100)
        .map(|i| (splitmix64(i as u64), 0, i as u64))
        .collect();

    let index = builder.build_from_flush(entries, 1).await.unwrap();

    // Batch lookup
    let keys: Vec<u64> = vec![10, 20, 30, 40, 50, 999]; // 999 doesn't exist
    let results = index.find_batch(&keys).await.unwrap();

    assert_eq!(results.len(), 6);

    // Check that we found the existing keys
    for (key, locations) in &results {
        if *key == 999 {
            assert!(locations.is_empty());
        } else {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].row_idx, *key);
        }
    }
}

#[tokio::test]
async fn test_empty_index() {
    let client = Arc::new(MockS3Client::new());
    let config = test_config();

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["empty.parquet".to_string()]);

    let index = builder.build_from_flush(vec![], 1).await.unwrap();

    // Lookups should return empty
    let locations = index.find(100).await.unwrap();
    assert!(locations.is_empty());

    // Metadata should be valid
    assert_eq!(index.num_entries().await.unwrap(), 0);
}

#[tokio::test]
async fn test_large_index() {
    let client = Arc::new(MockS3Client::new());
    let config = S3IndexConfig::new("test-bucket", "test/indices")
        .with_cache_size(1024 * 1024) // 1 MB
        .with_num_buckets(256);

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["data.parquet".to_string()]);

    // Create 10000 entries
    let entries: Vec<_> = (0..10000)
        .map(|i| (splitmix64(i as u64), 0, i as u64))
        .collect();

    let index = builder.build_from_flush(entries, 1).await.unwrap();

    assert_eq!(index.num_entries().await.unwrap(), 10000);

    // Verify some random lookups
    for key in [0, 100, 1000, 5000, 9999] {
        let locations = index.find(key).await.unwrap();
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].row_idx, key);
    }
}

#[tokio::test]
async fn test_duplicate_keys() {
    let client = Arc::new(MockS3Client::new());
    let config = test_config();

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["data.parquet".to_string()]);

    // Same key, different rows (shouldn't happen in practice with PKs, but test the behavior)
    let entries = vec![
        (splitmix64(100), 0, 0),
        (splitmix64(100), 0, 1), // Duplicate key
        (splitmix64(100), 0, 2), // Another duplicate
    ];

    let index = builder.build_from_flush(entries, 1).await.unwrap();

    // Should return all locations
    let locations = index.find(100).await.unwrap();
    assert_eq!(locations.len(), 3);
}

#[tokio::test]
async fn test_cache_behavior() {
    let client = Arc::new(MockS3Client::new());
    let config = S3IndexConfig::new("test-bucket", "test/indices")
        .with_cache_size(10 * 1024 * 1024)
        .with_num_buckets(8); // Few buckets to ensure cache hits

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["data.parquet".to_string()]);

    let entries: Vec<_> = (0..100)
        .map(|i| (splitmix64(i as u64), 0, i as u64))
        .collect();

    let index = builder.build_from_flush(entries, 1).await.unwrap();

    // First lookup - cache miss
    let _ = index.find(50).await.unwrap();

    let stats = index.cache_stats().await;
    assert!(stats.num_entries > 0);

    // Second lookup of same bucket - should be cache hit
    // (keys that hash to same bucket)
    let _ = index.find(50).await.unwrap();

    // Cache should still have same or more entries
    let stats2 = index.cache_stats().await;
    assert!(stats2.num_entries >= stats.num_entries);
}

#[tokio::test]
async fn test_index_metadata() {
    let client = Arc::new(MockS3Client::new());
    let config = test_config();

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["a.parquet".to_string(), "b.parquet".to_string()]);

    let entries = vec![
        (splitmix64(1), 0, 0),
        (splitmix64(2), 1, 0),
    ];

    let index = builder.build_from_flush(entries, 42).await.unwrap();

    assert_eq!(index.index_id(), 42);
    assert_eq!(index.num_buckets().await.unwrap(), 64);
    assert_eq!(index.num_entries().await.unwrap(), 2);

    let files = index.files().await.unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0], "a.parquet");
    assert_eq!(files[1], "b.parquet");
}

#[tokio::test]
async fn test_delete_index() {
    let client = Arc::new(MockS3Client::new());
    let config = test_config();

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["data.parquet".to_string()]);

    let entries = vec![(splitmix64(1), 0, 0)];
    let index = builder.build_from_flush(entries, 1).await.unwrap();

    let s3_key = index.s3_key().to_string();
    assert!(client.contains(&s3_key));

    index.delete().await.unwrap();
    assert!(!client.contains(&s3_key));
}

#[tokio::test]
async fn test_build_from_keys() {
    let client = Arc::new(MockS3Client::new());
    let config = test_config();

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["data.parquet".to_string()]);

    // Use raw keys instead of pre-hashed values
    let key_entries = vec![
        (100u64, 0u32, 0u64),
        (200, 0, 1),
        (300, 0, 2),
    ];

    let index = builder.build_from_keys(key_entries, 1).await.unwrap();

    // Lookups should work
    assert_eq!(index.find(100).await.unwrap().len(), 1);
    assert_eq!(index.find(200).await.unwrap().len(), 1);
    assert_eq!(index.find(300).await.unwrap().len(), 1);
}

#[tokio::test]
async fn test_index_sharing() {
    let client = Arc::new(MockS3Client::new());
    let config = test_config();

    let builder = S3GlobalIndexBuilder::new(client.clone(), config.clone())
        .set_files(vec!["data.parquet".to_string()]);

    let entries: Vec<_> = (0..100)
        .map(|i| (splitmix64(i as u64), 0, i as u64))
        .collect();

    let index = builder.build_from_flush(entries, 1).await.unwrap();

    // Clone the index (should share internal state)
    let index2 = index.clone();

    // Both should work and share cache
    let loc1 = &index.find(50).await.unwrap()[0];
    let loc2 = &index2.find(50).await.unwrap()[0];

    assert_eq!(loc1.row_idx, loc2.row_idx);
}
