#![cfg(feature = "import")]

use burn::module::{Module, ModuleMapper, Param, Quantizer};
use burn::prelude::{Backend, Tensor};
use burn::tensor::FloatDType;
use burn::tensor::quantization::{
    Calibration, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};

use super::load_policy::BpkPrecision;

pub fn apply_import_precision<B: Backend, M: Module<B>>(module: M, precision: BpkPrecision) -> M {
    match precision {
        BpkPrecision::F32 => module,
        BpkPrecision::F16 => cast_module_float_dtype(module, FloatDType::F16),
        BpkPrecision::Fp8 | BpkPrecision::Q4 => quantize_module(module, precision),
    }
}

fn quantize_module<B: Backend, M: Module<B>>(module: M, precision: BpkPrecision) -> M {
    let mut quantizer = Quantizer {
        calibration: Calibration::MinMax,
        scheme: quant_scheme(precision),
    };
    module.quantize_weights(&mut quantizer)
}

fn quant_scheme(precision: BpkPrecision) -> QuantScheme {
    match precision {
        BpkPrecision::Fp8 => QuantScheme::default()
            .with_value(QuantValue::Q8F)
            .with_level(QuantLevel::Tensor)
            .with_mode(QuantMode::Symmetric)
            .with_param(QuantParam::F32)
            .with_store(QuantStore::Native),
        BpkPrecision::Q4 => QuantScheme::default()
            .with_value(QuantValue::Q4F)
            .with_level(QuantLevel::Tensor)
            .with_mode(QuantMode::Symmetric)
            .with_param(QuantParam::F16)
            .with_store(QuantStore::U32),
        BpkPrecision::F32 | BpkPrecision::F16 => QuantScheme::default(),
    }
}

struct FloatDTypeMapper {
    dtype: FloatDType,
}

impl<B: Backend> ModuleMapper<B> for FloatDTypeMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = tensor.cast(self.dtype);
        Param::from_mapped_value(id, tensor, mapper)
    }
}

fn cast_module_float_dtype<B: Backend, M: Module<B>>(module: M, dtype: FloatDType) -> M {
    let mut mapper = FloatDTypeMapper { dtype };
    module.map(&mut mapper)
}

#[cfg(test)]
mod tests {
    use burn::tensor::quantization::{QuantLevel, QuantParam, QuantStore, QuantValue};

    use super::{BpkPrecision, quant_scheme};

    #[test]
    fn fp8_scheme_matches_expected_storage_and_params() {
        let scheme = quant_scheme(BpkPrecision::Fp8);
        assert_eq!(scheme.value, QuantValue::Q8F);
        assert_eq!(scheme.level, QuantLevel::Tensor);
        assert_eq!(scheme.param, QuantParam::F32);
        assert_eq!(scheme.store, QuantStore::Native);
    }

    #[test]
    fn q4_scheme_matches_expected_storage_and_params() {
        let scheme = quant_scheme(BpkPrecision::Q4);
        assert_eq!(scheme.value, QuantValue::Q4F);
        assert_eq!(scheme.level, QuantLevel::Tensor);
        assert_eq!(scheme.param, QuantParam::F16);
        assert_eq!(scheme.store, QuantStore::U32);
    }
}
