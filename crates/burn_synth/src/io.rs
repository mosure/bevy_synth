use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub enum ImageSource {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

impl ImageSource {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(bytes.into())
    }

    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Path(path) => Some(path.as_path()),
            Self::Bytes(_) => None,
        }
    }

    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Path(_) => None,
            Self::Bytes(bytes) => Some(bytes),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextPrompt(pub String);

impl From<String> for TextPrompt {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for TextPrompt {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl TextPrompt {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
