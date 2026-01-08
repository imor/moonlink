//! # Index Merge Configuration
//!
//! This module configures when and how file indices are merged to prevent fragmentation.
//!
//! ## The Fragmentation Problem
//!
//! As data flushes to disk, small index files accumulate:
//!
//! ```text
//! After 20 flushes: [index_1: 10KB, index_2: 12KB, ..., index_20: 11KB]
//! ```
//!
//! **Problems:**
//! - Lookups must check all indices sequentially
//! - More file handles and cache pressure
//! - Slower batch lookups
//!
//! ## Solution: Index Merging
//!
//! Periodically merge small indices into larger ones:
//!
//! ```text
//! Before merge:
//! [index_1: 10KB, index_2: 12KB, index_3: 8KB, index_4: 11KB]
//!     │
//!     ▼ Merge!
//!     │
//! After merge:
//! [index_merged: 41KB]
//! ```
//!
//! ## When Merging Happens
//!
//! A merge triggers when at least `min_file_indices_to_merge` small indices exist
//! (below `index_block_final_size`). At most `max_file_indices_to_merge` are merged
//! at once to limit resource usage.
//!
//! ## Configuration Parameters
//!
//! ### `min_file_indices_to_merge`
//!
//! Minimum number of small indices needed before triggering a merge.
//!
//! **Example**: If set to 4:
//! ```text
//! 3 small indices: [10KB, 12KB, 8KB] → No merge yet
//! 4 small indices: [10KB, 12KB, 8KB, 11KB] → Merge triggered!
//! ```
//!
//! **Defaults**:
//! - Test builds: `u32::MAX` (disabled)
//! - Debug builds: `4`
//! - Release builds: `16`
//!
//! ### `max_file_indices_to_merge`
//!
//! Maximum number of indices to merge in one operation (to avoid huge merge operations).
//!
//! **Example**: If set to 8 and there are 10 small indices:
//! ```text
//! Merge only the first 8: [i1, i2, i3, i4, i5, i6, i7, i8]
//! Leave the rest: [i9, i10] ← Will be merged in next round
//! ```
//!
//! **Defaults**:
//! - Test builds: `u32::MAX` (disabled)
//! - Debug builds: `8`
//! - Release builds: `32`
//!
//! ### `index_block_final_size`
//!
//! Size threshold (in bytes) above which an index is considered "final" and won't be merged.
//!
//! **Example**: If set to 512 MiB:
//! ```text
//! Indices: [10MB, 20MB, 600MB, 15MB]
//!                        ↑
//!                      Final - skip this one!
//!                      
//! Only merge: [10MB, 20MB, 15MB] → 45MB merged index
//! Keep separate: [600MB] ← Already large enough
//! ```
//!
//! **Defaults**:
//! - Test builds: `u64::MAX` (disabled)
//! - Debug builds: `1 KiB` (very aggressive for testing)
//! - Release builds: `512 MiB`
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! // Use default configuration
//! let config = FileIndexMergeConfig::default();
//!
//! // Custom configuration for aggressive merging
//! let config = FileIndexMergeConfig::builder()
//!     .min_file_indices_to_merge(2)  // Merge as soon as we have 2
//!     .max_file_indices_to_merge(10) // Merge up to 10 at once
//!     .index_block_final_size(1 << 30)  // 1 GiB final size
//!     .build();
//!
//! // Completely disable merging
//! let config = FileIndexMergeConfig::disabled();
//! ```
//!
//! ## Merge Decision Logic
//!
//! ```rust,ignore
//! fn should_merge(indices: &[GlobalIndex], config: &FileIndexMergeConfig) -> bool {
//!     // Count indices below final size
//!     let small_indices: Vec<_> = indices.iter()
//!         .filter(|idx| idx.get_index_blocks_size() < config.index_block_final_size)
//!         .collect();
//!         
//!     // Trigger merge if we have enough small indices
//!     small_indices.len() >= config.min_file_indices_to_merge as usize
//! }
//!
//! fn merge_indices(indices: &mut Vec<GlobalIndex>, config: &FileIndexMergeConfig) {
//!     // Select small indices
//!     let to_merge: Vec<_> = indices.iter()
//!         .filter(|idx| idx.get_index_blocks_size() < config.index_block_final_size)
//!         .take(config.max_file_indices_to_merge as usize)
//!         .collect();
//!         
//!     // Merge them into one large index
//!     let merged = GlobalIndexBuilder::build_from_merge(&to_merge).await?;
//!     
//!     // Replace old indices with merged one
//!     // ... (remove to_merge, add merged)
//! }
//! ```

use more_asserts as ma;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

/// Configuration controlling when and how file indices are merged.
///
/// Merging prevents fragmentation by combining many small index files into
/// fewer large ones, improving lookup performance.
///
/// # Fields
///
/// - `min_file_indices_to_merge`: Minimum count to trigger merge
/// - `max_file_indices_to_merge`: Maximum count to merge at once
/// - `index_block_final_size`: Size threshold for "final" indices
///
/// # Example
///
/// ```rust,ignore
/// let config = FileIndexMergeConfig::builder()
///     .min_file_indices_to_merge(4)
///     .max_file_indices_to_merge(8)
///     .index_block_final_size(512 * 1024 * 1024) // 512 MiB
///     .build();
///
/// config.validate(); // Ensures min <= max
/// ```
///
/// # Defaults
///
/// Different defaults apply based on build type:
///
/// | Parameter | Test | Debug | Release |
/// |-----------|------|-------|---------|
/// | `min_file_indices_to_merge` | MAX | 4 | 16 |
/// | `max_file_indices_to_merge` | MAX | 8 | 32 |
/// | `index_block_final_size` | MAX | 1 KiB | 512 MiB |
///
/// Test builds disable merging by default to simplify testing.
#[derive(Clone, Debug, PartialEq, TypedBuilder, Deserialize, Serialize)]
pub struct FileIndexMergeConfig {
    /// Number of small index files needed to trigger a merge.
    ///
    /// Only indices smaller than `index_block_final_size` are counted.
    ///
    /// # Example
    ///
    /// If set to 4:
    /// ```text
    /// Indices: [10KB, 15KB, 12KB] → 3 small indices, no merge
    /// Add one more: [10KB, 15KB, 12KB, 14KB] → 4 small indices, merge!
    /// ```
    #[serde(default = "FileIndexMergeConfig::default_min_file_indices_to_merge")]
    pub min_file_indices_to_merge: u32,

    /// Maximum number of indices to merge in a single operation.
    ///
    /// Prevents merging too many files at once, which could:
    /// - Use too much memory
    /// - Take too long
    /// - Create locks for extended periods
    ///
    /// # Example
    ///
    /// If set to 8 with 12 small indices:
    /// ```text
    /// First merge: Combine indices [0..8] → 1 merged index
    /// Result: [merged_1, idx_8, idx_9, idx_10, idx_11]
    ///
    /// Later (after more flushes):
    /// Second merge: Combine [merged_1, idx_8, idx_9, idx_10, ...]
    /// ```
    #[serde(default = "FileIndexMergeConfig::default_max_file_indices_to_merge")]
    pub max_file_indices_to_merge: u32,

    /// Size in bytes above which an index is considered final.
    ///
    /// Final indices are not merged, even if there are many of them.
    /// This prevents repeatedly re-merging already-large indices.
    ///
    /// # Example
    ///
    /// If set to 100 MiB:
    /// ```text
    /// Indices and sizes:
    /// - index_1: 10 MiB  ← Small, can merge
    /// - index_2: 150 MiB ← Final, skip
    /// - index_3: 20 MiB  ← Small, can merge  
    /// - index_4: 200 MiB ← Final, skip
    /// - index_5: 15 MiB  ← Small, can merge
    ///
    /// Only merge: [index_1, index_3, index_5] = 45 MiB
    /// Keep unchanged: [index_2: 150 MiB, index_4: 200 MiB]
    /// ```
    #[serde(default = "FileIndexMergeConfig::default_index_block_final_size")]
    pub index_block_final_size: u64,
}

impl FileIndexMergeConfig {
    // Default constants for different build configurations

    /// Default minimum indices to merge (test builds - disabled)
    #[cfg(test)]
    pub const DEFAULT_MIN_FILE_INDICES_TO_MERGE: u32 = u32::MAX;

    /// Default maximum indices to merge (test builds - disabled)
    #[cfg(test)]
    pub const DEFAULT_MAX_FILE_INDICES_TO_MERGE: u32 = u32::MAX;

    /// Default final size threshold (test builds - disabled)
    #[cfg(test)]
    pub const DEFAULT_INDEX_BLOCK_FINAL_SIZE: u64 = u64::MAX;

    /// Default minimum indices to merge (debug builds)
    #[cfg(all(not(test), debug_assertions))]
    pub const DEFAULT_MIN_FILE_INDICES_TO_MERGE: u32 = 4;

    /// Default maximum indices to merge (debug builds)
    #[cfg(all(not(test), debug_assertions))]
    pub const DEFAULT_MAX_FILE_INDICES_TO_MERGE: u32 = 8;

    /// Default final size threshold (debug builds - 1 KiB for aggressive testing)
    #[cfg(all(not(test), debug_assertions))]
    pub const DEFAULT_INDEX_BLOCK_FINAL_SIZE: u64 = 1 << 10; // 1KiB

    /// Default minimum indices to merge (release builds)
    #[cfg(all(not(test), not(debug_assertions)))]
    pub const DEFAULT_MIN_FILE_INDICES_TO_MERGE: u32 = 16;

    /// Default maximum indices to merge (release builds)
    #[cfg(all(not(test), not(debug_assertions)))]
    pub const DEFAULT_MAX_FILE_INDICES_TO_MERGE: u32 = 32;

    /// Default final size threshold (release builds - 512 MiB)
    #[cfg(all(not(test), not(debug_assertions)))]
    pub const DEFAULT_INDEX_BLOCK_FINAL_SIZE: u64 = 1 << 29; // 512MiB

    /// Get the default minimum file indices to merge for serde deserialization.
    pub fn default_min_file_indices_to_merge() -> u32 {
        Self::DEFAULT_MIN_FILE_INDICES_TO_MERGE
    }

    /// Get the default maximum file indices to merge for serde deserialization.
    pub fn default_max_file_indices_to_merge() -> u32 {
        Self::DEFAULT_MAX_FILE_INDICES_TO_MERGE
    }

    /// Get the default index block final size for serde deserialization.
    pub fn default_index_block_final_size() -> u64 {
        Self::DEFAULT_INDEX_BLOCK_FINAL_SIZE
    }

    /// Validate that configuration values are consistent.
    ///
    /// # Panics
    ///
    /// Panics if `min_file_indices_to_merge > max_file_indices_to_merge`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let valid = FileIndexMergeConfig::builder()
    ///     .min_file_indices_to_merge(4)
    ///     .max_file_indices_to_merge(8)
    ///     .index_block_final_size(1 << 20)
    ///     .build();
    /// valid.validate(); // OK
    ///
    /// let invalid = FileIndexMergeConfig::builder()
    ///     .min_file_indices_to_merge(10)  // min > max!
    ///     .max_file_indices_to_merge(5)
    ///     .index_block_final_size(1 << 20)
    ///     .build();
    /// invalid.validate(); // Panics!
    /// ```
    pub fn validate(&self) {
        ma::assert_le!(
            self.min_file_indices_to_merge,
            self.max_file_indices_to_merge
        );
    }
}

impl Default for FileIndexMergeConfig {
    /// Create configuration with build-appropriate defaults.
    ///
    /// See struct documentation for default values per build type.
    fn default() -> Self {
        Self {
            min_file_indices_to_merge: Self::DEFAULT_MIN_FILE_INDICES_TO_MERGE,
            max_file_indices_to_merge: Self::DEFAULT_MAX_FILE_INDICES_TO_MERGE,
            index_block_final_size: Self::DEFAULT_INDEX_BLOCK_FINAL_SIZE,
        }
    }
}

impl FileIndexMergeConfig {
    /// Create a configuration with index merging enabled.
    ///
    /// Uses the default values for the current build type.
    /// Same as calling `FileIndexMergeConfig::default()`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = FileIndexMergeConfig::enabled();
    /// // In release builds:
    /// assert_eq!(config.min_file_indices_to_merge, 16);
    /// assert_eq!(config.max_file_indices_to_merge, 32);
    /// assert_eq!(config.index_block_final_size, 512 * 1024 * 1024);
    /// ```
    pub fn enabled() -> Self {
        Self::default()
    }

    /// Create a configuration with index merging completely disabled.
    ///
    /// Sets all thresholds to maximum values, preventing any merges from occurring.
    /// Useful for:
    /// - Testing specific scenarios without merge interference
    /// - Debugging merge-related issues
    /// - Temporary performance testing
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = FileIndexMergeConfig::disabled();
    /// assert_eq!(config.min_file_indices_to_merge, u32::MAX);
    /// assert_eq!(config.max_file_indices_to_merge, u32::MAX);
    /// assert_eq!(config.index_block_final_size, u64::MAX);
    ///
    /// // With this config, merges will never be triggered
    /// // even with thousands of tiny index files
    /// ```
    pub fn disabled() -> Self {
        Self {
            min_file_indices_to_merge: u32::MAX,
            max_file_indices_to_merge: u32::MAX,
            index_block_final_size: u64::MAX,
        }
    }
}
