pub type SceneResult<T> = Result<T, SceneError>;

#[derive(Debug)]
pub enum SceneError {
    Config(String),
    Io(String),
    Image(String),
    Http(String),
    Provider(String),
    Parse(String),
    Validation(String),
}

impl std::fmt::Display for SceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(err) => write!(f, "configuration error: {err}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Image(err) => write!(f, "image error: {err}"),
            Self::Http(err) => write!(f, "OpenAI HTTP error: {err}"),
            Self::Provider(err) => write!(f, "provider error: {err}"),
            Self::Parse(err) => write!(f, "parse error: {err}"),
            Self::Validation(err) => write!(f, "validation error: {err}"),
        }
    }
}

impl std::error::Error for SceneError {}

impl From<std::io::Error> for SceneError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<image::ImageError> for SceneError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value.to_string())
    }
}
