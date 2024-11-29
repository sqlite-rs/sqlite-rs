use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SqliteError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid file header")]
    InvalidHeader,
    #[error("Unsupported file format version")]
    UnsupportedVersion,
    #[error("Invalid page type")]
    InvalidPageType,
    #[error("Invalid cell content")]
    InvalidCellContent,
    #[error("SQL parse error: {0}")]
    SqlParseError(String),
    #[error("Table not found: {0}")]
    TableNotFound(String),
}
