use super::sampling_kernels::*;
use super::*;

pub fn dense_trilinear_sample_attrs_wgpu(
    positions: BurnTensor<DefaultWgpuBackend, 2>,
    occupancy: BurnTensor<DefaultWgpuBackend, 1, Int>,
    attrs: BurnTensor<DefaultWgpuBackend, 2>,
    spatial: [usize; 3],
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    let [rows, pos_cols] = positions.dims();
    if pos_cols != 3 {
        return Err(format!(
            "dense trilinear positions tensor must have 3 columns, got {pos_cols}"
        ));
    }
    let [cells, attr_cols] = attrs.dims();
    if attr_cols != 6 {
        return Err(format!(
            "dense trilinear attrs tensor must have 6 columns, got {attr_cols}"
        ));
    }
    let [occupancy_len] = occupancy.dims();
    let expected_cells = spatial[0]
        .checked_mul(spatial[1])
        .and_then(|value| value.checked_mul(spatial[2]))
        .ok_or_else(|| {
            format!(
                "dense trilinear spatial volume overflow: [{},{},{}]",
                spatial[0], spatial[1], spatial[2]
            )
        })?;
    if expected_cells == 0 {
        return Err("dense trilinear sampling requires non-empty spatial volume".to_string());
    }
    if cells != expected_cells || occupancy_len != expected_cells {
        return Err(format!(
            "dense trilinear tensor length mismatch: attrs_rows={} occupancy_len={} expected_cells={}",
            cells, occupancy_len, expected_cells
        ));
    }
    if rows == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2>::zeros(
            [0, 7],
            &positions.device(),
        ));
    }

    let max_x = i32::try_from(spatial[0].saturating_sub(1))
        .map_err(|_| format!("dense trilinear spatial x={} exceeds i32 range", spatial[0]))?;
    let max_y = i32::try_from(spatial[1].saturating_sub(1))
        .map_err(|_| format!("dense trilinear spatial y={} exceeds i32 range", spatial[1]))?;
    let max_z = i32::try_from(spatial[2].saturating_sub(1))
        .map_err(|_| format!("dense trilinear spatial z={} exceeds i32 range", spatial[2]))?;
    let stride_x = i32::try_from(spatial[0])
        .map_err(|_| format!("dense trilinear spatial x={} exceeds i32 range", spatial[0]))?;
    let stride_xy_u64 = (spatial[0] as u64)
        .checked_mul(spatial[1] as u64)
        .ok_or_else(|| {
            format!(
                "dense trilinear stride overflow for spatial=[{},{},{}]",
                spatial[0], spatial[1], spatial[2]
            )
        })?;
    let stride_xy = i32::try_from(stride_xy_u64).map_err(|_| {
        format!(
            "dense trilinear stride_xy={} exceeds i32 range",
            stride_xy_u64
        )
    })?;

    let positions_p = positions.reshape([rows * 3]).into_primitive().tensor();
    let occupancy_p = occupancy.into_primitive();
    let attrs_p = attrs.reshape([cells * 6]).into_primitive().tensor();
    let output_elements = rows
        .checked_mul(7)
        .ok_or_else(|| "dense trilinear output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "dense trilinear output byte size overflow".to_string())?;

    let output = CubeTensor::new_contiguous(
        positions_p.client.clone(),
        positions_p.device.clone(),
        Shape::new([rows, 7]),
        positions_p.client.empty(output_bytes),
        DType::F32,
    );
    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(&positions_p.client, rows, cube_dim);
    let dim_x = spatial[0] as f32;
    let dim_y = spatial[1] as f32;
    let dim_z = spatial[2] as f32;
    unsafe {
        dense_trilinear_sample_attrs_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &positions_p.client,
            cube_count,
            cube_dim,
            positions_p.clone().into_array_arg(),
            occupancy_p.clone().into_array_arg(),
            attrs_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            dim_x,
            dim_y,
            dim_z,
            max_x,
            max_y,
            max_z,
            stride_x,
            stride_xy,
        )
        .map_err(|err| format!("dense_trilinear_sample_attrs_kernel launch failed: {err:?}"))?;
    }

    Ok(BurnTensor::from_primitive(TensorPrimitive::Float(output)))
}
