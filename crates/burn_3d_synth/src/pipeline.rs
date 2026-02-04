use crate::io::{ImageSource, TextPrompt};
use crate::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct PipelineInput {
    pub image: Option<ImageSource>,
    pub text: Option<TextPrompt>,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct PipelineOutput<M = Mesh> {
    pub mesh: Option<M>,
}

pub type MeshOutput = PipelineOutput<Mesh>;
