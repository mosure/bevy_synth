use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::{Backend, Tensor};

use super::load_policy::{BpkPrecisionPreference, BurnpackLoadPolicy};

struct DequantizeMapper;

impl<B: Backend> ModuleMapper<B> for DequantizeMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor.dequantize(), mapper)
    }
}

fn dequantize_module_weights<B: Backend, M: Module<B>>(module: M) -> M {
    let mut mapper = DequantizeMapper;
    module.map(&mut mapper)
}

/// Quantized fp8/q4 artifacts currently require selective dequantization on load because
/// some cubecl quantized tensor ops used by TripoSG are not implemented in Burn 0.19.
pub fn maybe_postprocess_loaded_module<B: Backend, M: Module<B>>(
    module: M,
    policy: BurnpackLoadPolicy,
) -> M {
    match policy.precision {
        BpkPrecisionPreference::PreferFp8 | BpkPrecisionPreference::PreferQ4 => {
            dequantize_module_weights(module)
        }
        BpkPrecisionPreference::PreferF16 | BpkPrecisionPreference::PreferF32 => module,
    }
}
