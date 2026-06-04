use crate::io::{ImageSource, TextPrompt};
use crate::mesh::Mesh;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SynthesisModel {
    Triposg,
    Trellis,
    Triposplat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ForegroundModel {
    Rmbg14,
    Rmbg2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelSelection {
    pub synthesis_models: Vec<SynthesisModel>,
    pub foreground_model: ForegroundModel,
}

impl Default for ModelSelection {
    fn default() -> Self {
        Self {
            synthesis_models: vec![SynthesisModel::Triposg],
            foreground_model: ForegroundModel::Rmbg14,
        }
    }
}

impl ModelSelection {
    pub fn new(
        synthesis_models: impl IntoIterator<Item = SynthesisModel>,
        foreground_model: ForegroundModel,
    ) -> Self {
        Self {
            synthesis_models: sanitize_synthesis_models(synthesis_models),
            foreground_model,
        }
    }

    pub fn supports_synthesis_model(&self, model: SynthesisModel) -> bool {
        self.synthesis_models.contains(&model)
    }
}

pub fn sanitize_synthesis_models(
    models: impl IntoIterator<Item = SynthesisModel>,
) -> Vec<SynthesisModel> {
    let mut out = Vec::new();
    for model in models {
        if !out.contains(&model) {
            out.push(model);
        }
    }
    if out.is_empty() {
        out.push(SynthesisModel::Triposg);
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct PipelineInput {
    pub image: Option<ImageSource>,
    pub text: Option<TextPrompt>,
    pub seed: Option<u64>,
    pub model_selection: ModelSelection,
}

#[derive(Clone, Debug, Default)]
pub struct PipelineOutput<M = Mesh> {
    pub mesh: Option<M>,
    pub asset: Option<SynthesisAsset>,
}

pub type MeshOutput = PipelineOutput<Mesh>;

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum SynthesisAsset {
    Mesh(Mesh),
    GaussianSplat(burn_triposplat::GaussianSplatCloud),
}

#[cfg(test)]
mod tests {
    use super::{ForegroundModel, ModelSelection, SynthesisModel, sanitize_synthesis_models};

    #[test]
    fn sanitize_defaults_to_triposg_when_empty() {
        let models = sanitize_synthesis_models([]);
        assert_eq!(models, vec![SynthesisModel::Triposg]);
    }

    #[test]
    fn sanitize_deduplicates_models_preserving_order() {
        let models = sanitize_synthesis_models([
            SynthesisModel::Triposg,
            SynthesisModel::Triposg,
            SynthesisModel::Trellis,
            SynthesisModel::Trellis,
            SynthesisModel::Triposplat,
            SynthesisModel::Triposplat,
        ]);
        assert_eq!(
            models,
            vec![
                SynthesisModel::Triposg,
                SynthesisModel::Trellis,
                SynthesisModel::Triposplat
            ]
        );
    }

    #[test]
    fn model_selection_normalizes_synthesis_models() {
        let selection = ModelSelection::new(
            [
                SynthesisModel::Trellis,
                SynthesisModel::Triposg,
                SynthesisModel::Triposg,
            ],
            ForegroundModel::Rmbg14,
        );
        assert_eq!(
            selection.synthesis_models,
            vec![SynthesisModel::Trellis, SynthesisModel::Triposg]
        );
        assert_eq!(selection.foreground_model, ForegroundModel::Rmbg14);
    }

    #[test]
    fn model_selection_default_prefers_rmbg14() {
        let selection = ModelSelection::default();
        assert_eq!(selection.foreground_model, ForegroundModel::Rmbg14);
    }
}
