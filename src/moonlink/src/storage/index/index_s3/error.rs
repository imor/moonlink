//! Error types for S3 index operations.
//!
//! This module defines error types specific to S3-based index operations,
//! separate from the main moonlink error types.

use std::fmt;

/// Result type for S3 index operations.
pub type S3IndexResult<T> = std::result::Result<T, S3IndexError>;

/// Errors that can occur during S3 index operations.
#[derive(Debug)]
pub enum S3IndexError {
    /// S3 object was not found.
    ObjectNotFound {
        /// The S3 key that was not found.
        key: String,
    },

    /// S3 operation failed.
    S3Operation {
        /// Description of the operation that failed.
        operation: String,
        /// The S3 key involved.
        key: String,
        /// The underlying error message.
        message: String,
    },

    /// Invalid byte range requested.
    InvalidRange {
        /// The S3 key.
        key: String,
        /// Requested range start.
        start: u64,
        /// Requested range end.
        end: u64,
        /// Object size.
        object_size: u64,
    },

    /// Index file format is invalid or corrupted.
    InvalidFormat {
        /// Description of the format error.
        message: String,
    },

    /// Magic number doesn't match expected value.
    InvalidMagicNumber {
        /// Expected magic number.
        expected: [u8; 8],
        /// Actual magic number found.
        found: [u8; 8],
    },

    /// Index version is not supported.
    UnsupportedVersion {
        /// The version found in the index.
        version: u32,
        /// Maximum supported version.
        max_supported: u32,
    },

    /// Checksum validation failed.
    ChecksumMismatch {
        /// Expected checksum.
        expected: u64,
        /// Actual checksum computed.
        actual: u64,
    },

    /// Entry decoding failed.
    EntryDecodeError {
        /// Description of the decoding error.
        message: String,
    },

    /// IO error during index operations.
    Io(std::io::Error),

    /// Serialization/deserialization error.
    SerdeError {
        /// Description of the error.
        message: String,
    },

    /// Cache error.
    CacheError {
        /// Description of the cache error.
        message: String,
    },

    /// Generic internal error.
    Internal {
        /// Error message.
        message: String,
    },
}

impl fmt::Display for S3IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            S3IndexError::ObjectNotFound { key } => {
                write!(f, "S3 object not found: {}", key)
            }
            S3IndexError::S3Operation {
                operation,
                key,
                message,
            } => {
                write!(f, "S3 {} failed for '{}': {}", operation, key, message)
            }
            S3IndexError::InvalidRange {
                key,
                start,
                end,
                object_size,
            } => {
                write!(
                    f,
                    "Invalid range {}-{} for object '{}' with size {}",
                    start, end, key, object_size
                )
            }
            S3IndexError::InvalidFormat { message } => {
                write!(f, "Invalid index format: {}", message)
            }
            S3IndexError::InvalidMagicNumber { expected, found } => {
                write!(
                    f,
                    "Invalid magic number: expected {:?}, found {:?}",
                    expected, found
                )
            }
            S3IndexError::UnsupportedVersion {
                version,
                max_supported,
            } => {
                write!(
                    f,
                    "Unsupported index version {}, max supported is {}",
                    version, max_supported
                )
            }
            S3IndexError::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "Checksum mismatch: expected {:#x}, got {:#x}",
                    expected, actual
                )
            }
            S3IndexError::EntryDecodeError { message } => {
                write!(f, "Entry decode error: {}", message)
            }
            S3IndexError::Io(e) => {
                write!(f, "IO error: {}", e)
            }
            S3IndexError::SerdeError { message } => {
                write!(f, "Serialization error: {}", message)
            }
            S3IndexError::CacheError { message } => {
                write!(f, "Cache error: {}", message)
            }
            S3IndexError::Internal { message } => {
                write!(f, "Internal error: {}", message)
            }
        }
    }
}

impl std::error::Error for S3IndexError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            S3IndexError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for S3IndexError {
    fn from(e: std::io::Error) -> Self {
        S3IndexError::Io(e)
    }
}

impl From<serde_json::Error> for S3IndexError {
    fn from(e: serde_json::Error) -> Self {
        S3IndexError::SerdeError {
            message: e.to_string(),
        }
    }
}

impl S3IndexError {
    /// Create an S3 operation error.
    pub fn s3_operation(
        operation: impl Into<String>,
        key: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        S3IndexError::S3Operation {
            operation: operation.into(),
            key: key.into(),
            message: message.into(),
        }
    }

    /// Create an invalid format error.
    pub fn invalid_format(message: impl Into<String>) -> Self {
        S3IndexError::InvalidFormat {
            message: message.into(),
        }
    }

    /// Create an internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        S3IndexError::Internal {
            message: message.into(),
        }
    }
}
