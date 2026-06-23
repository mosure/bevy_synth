use burn::tensor::{Tensor, TensorPrimitive};
use cubek::matmul::{
    definition::{MatmulElems, MatmulGlobalElems},
    launch::Strategy,
};
use cubek::std::InputBinding;

pub(crate) type TrellisWgpuBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

pub(crate) fn matmul_2d_naive(
    lhs: Tensor<TrellisWgpuBackend, 2>,
    rhs: Tensor<TrellisWgpuBackend, 2>,
    context: &str,
) -> Tensor<TrellisWgpuBackend, 2> {
    let [rows, lhs_k] = lhs.dims();
    let [rhs_k, cols] = rhs.dims();
    if lhs_k != rhs_k {
        panic!("{context}: matmul shape mismatch lhs=[{rows},{lhs_k}] rhs=[{rhs_k},{cols}]");
    }
    if rows == 0 || cols == 0 {
        return Tensor::<TrellisWgpuBackend, 2>::zeros([rows, cols], &lhs.device());
    }

    let out_dtype: burn::tensor::FloatDType = lhs.dtype().into();
    let device = lhs.device();
    let out = Tensor::<TrellisWgpuBackend, 2>::zeros([rows, cols], &device).cast(out_dtype);

    let lhs = lhs.into_primitive().tensor();
    let rhs = rhs.into_primitive().tensor();
    let out = out.into_primitive().tensor();

    let lhs_binding = InputBinding::new(lhs.clone().binding(), lhs.dtype.into());
    let rhs_binding = InputBinding::new(rhs.clone().binding(), rhs.dtype.into());
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: lhs.dtype.into(),
        rhs: rhs.dtype.into(),
        out: out.dtype.into(),
    });

    cubek::matmul::launch::launch_ref(
        &Strategy::Naive,
        &lhs.client,
        lhs_binding,
        rhs_binding,
        out.clone().binding(),
        &mut dtypes,
    )
    .unwrap_or_else(|err| panic!("{context}: wasm-safe naive matmul failed: {err:?}"));

    Tensor::from_primitive(TensorPrimitive::Float(out))
}
