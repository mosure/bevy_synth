use cubecl::{
    client::ComputeClient,
    prelude::CubePrimitive,
    {CubeDim, Runtime},
};
use cubek_matmul::{
    components::{global::PartitionedStageFamily, stage::StridedStageFamily},
    routines::find_instruction_size,
};

use crate::definition::{
    AttentionBlueprint, AttentionElems, AttentionPartitionSize, AttentionProblem,
    AttentionSetupError, AttentionStageSize, AttentionTilingScheme, HypercubeBlueprint,
};
use crate::{
    components::stage::plane::PlanePartitionStageAttentionFamily,
    components::tile::TileAttentionKind, definition::AttentionTileSize,
};
use crate::{
    components::{
        batch::simple::SimpleBatchAttentionFamily, global::simple::SimpleGlobalAttentionFamily,
    },
    routines::Routine,
};
use crate::{
    launch::BlueprintStrategy,
    routines::{DeviceSettings, LaunchInfo},
};

#[derive(Debug, Clone)]
pub struct BlackboxAcceleratedRoutine {}

#[derive(Debug, Clone)]
pub struct BlackboxAcceleratedStrategy {
    pub num_planes: u8,
    pub seq_q: u8,
    pub seq_kv: u8,
}

const WGPU_BLACKBOX_MAX_VERIFIED_SEQ_KV_FOR_HEAD_DIM_64: usize = 21_504;
const WGPU_BLACKBOX_TRELLIS_SLAT_MIN_LONG_SEQ: usize = 8_192;
const WGPU_BLACKBOX_MAX_VERIFIED_SEQ_Q_FOR_LONG_K_HEAD_DIM_128: usize = 12_288;
const WGPU_BLACKBOX_MAX_VERIFIED_SEQ_Q_FOR_LONG_K_HEAD_DIM_64: usize = 12_288;

impl Routine for BlackboxAcceleratedRoutine {
    const TILE_KIND: TileAttentionKind = TileAttentionKind::BlackboxAccelerated;

    type StageAttention = PlanePartitionStageAttentionFamily<
        StridedStageFamily,
        StridedStageFamily,
        PartitionedStageFamily,
    >;
    type GlobalAttention = SimpleGlobalAttentionFamily<Self::StageAttention>;
    type BatchAttention = SimpleBatchAttentionFamily<Self::GlobalAttention>;

    type Strategy = BlackboxAcceleratedStrategy;
    type Blueprint = AttentionBlueprint;

    fn prepare<R: Runtime>(
        problem: &AttentionProblem,
        device_settings: &DeviceSettings<R>,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<LaunchInfo<Self::Blueprint>, AttentionSetupError> {
        let tile_type = half::f16::as_type_native_unchecked().storage_type();
        let dtypes = AttentionElems::from_global_types(
            &problem.global_dtypes,
            tile_type,
            &problem.options.accumulator_precision,
        );

        let blueprint = blueprint(problem, device_settings, &dtypes, strategy)?;

        let num_planes = blueprint.tiling_scheme.stage_size.seq_q;
        let cube_dim = CubeDim::new_2d(blueprint.plane_dim, num_planes);

        let cube_count_plan =
            blueprint.cube_count_plan(&problem.dims, &device_settings.max_cube_count);

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan,
            address_type: problem.address_type,
        })
    }
}

fn blueprint<R: Runtime>(
    problem: &AttentionProblem,
    device: &DeviceSettings<R>,
    dtypes: &AttentionElems,
    strategy: BlueprintStrategy<BlackboxAcceleratedRoutine>,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    match strategy {
        BlueprintStrategy::Forced(attention_blueprint) => {
            validate::<R>(problem, attention_blueprint)
        }
        BlueprintStrategy::Inferred(strategy) => {
            let is_supported = |client: &ComputeClient<R>, mma| {
                client.properties().features.matmul.cmma.contains(&mma)
            };

            let supported_sizes = |client: &ComputeClient<R>, lhs_ty, rhs_ty, acc_ty| {
                client
                    .properties()
                    .features
                    .matmul
                    .cmma
                    .iter()
                    .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
                    .map(|it| (it.m, it.n, it.k).into())
                    .collect::<Vec<_>>()
            };
            let map_err = |err| {
                AttentionSetupError::Unavailable(
                    crate::definition::AttentionAvailabilityError::MatmulInstructionUnavailable(
                        err,
                    ),
                )
            };

            let tile_size_score_matmul = find_instruction_size::<R, _, _>(
                &device.client,
                (dtypes.query_tile, dtypes.key_value_tile, dtypes.softmax_acc),
                (
                    problem.dims.seq_q,
                    problem.dims.seq_kv,
                    problem.dims.head_dim,
                )
                    .into(),
                (None, None, None),
                is_supported,
                supported_sizes,
            )
            .map_err(map_err)?;

            let values_matmul = find_instruction_size::<R, _, _>(
                &device.client,
                (
                    dtypes.softmax_lhs,
                    dtypes.key_value_tile,
                    dtypes.accumulator,
                ),
                (
                    problem.dims.seq_q,
                    problem.dims.val_dim,
                    problem.dims.seq_kv,
                )
                    .into(),
                (
                    Some(tile_size_score_matmul.m),
                    None,
                    Some(tile_size_score_matmul.n),
                ),
                is_supported,
                supported_sizes,
            )
            .map_err(map_err)?;

            if tile_size_score_matmul.m != values_matmul.m {
                return Err(AttentionSetupError::InvalidConfig(Box::new(
                    "Seq_q mismatch: `m` of score_matmul does not match `m` of values_matmul. ",
                )));
            }

            if tile_size_score_matmul.n != values_matmul.k {
                return Err(AttentionSetupError::InvalidConfig(Box::new(
                    "Seq_kv mismatch: `n` of score_matmul does not match `k` of values_matmul. ",
                )));
            }

            let strategy = blackbox_strategy_for_problem(problem, strategy);
            let tile_size = AttentionTileSize {
                seq_q: tile_size_score_matmul.m,
                head_dim: tile_size_score_matmul.k,
                seq_kv: tile_size_score_matmul.n,
                val_dim: values_matmul.n,
            };

            let partition_head_dim = problem.dims.head_dim as u32 / tile_size.head_dim;
            let partition_val_dim = problem.dims.val_dim as u32 / tile_size.val_dim;

            let tiling_scheme = AttentionTilingScheme {
                tile_size,
                partition_size: AttentionPartitionSize {
                    seq_q: strategy.seq_q as u32,
                    head_dim: partition_head_dim,
                    seq_kv: strategy.seq_kv as u32,
                    val_dim: partition_val_dim,
                },
                stage_size: AttentionStageSize {
                    seq_q: strategy.num_planes as u32,
                },
            };

            let blueprint = AttentionBlueprint {
                hypercube_blueprint: HypercubeBlueprint::builder().build(),
                plane_dim: device.plane_dim,
                two_rows_in_array_tile: false,
                vector_sizes: device.vector_sizes.clone(),
                masked: problem.masked,
                causal: problem.options.causal,
                tiling_scheme,
                check_bounds: tiling_scheme.check_bounds(&problem.dims),
            };

            maybe_log_strategy(problem, &strategy, tile_size, &blueprint);

            validate::<R>(problem, blueprint)
        }
    }
}

fn validate<R: Runtime>(
    problem: &AttentionProblem,
    blueprint: AttentionBlueprint,
) -> Result<AttentionBlueprint, AttentionSetupError> {
    if wgpu_blackbox_long_k_shape_is_unsafe::<R>(problem) {
        return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
            "CubeK blackbox attention is disabled for unverified long-K WGPU shapes (head_dim=64 with batch_heads >= 16 and seq_kv > {WGPU_BLACKBOX_MAX_VERIFIED_SEQ_KV_FOR_HEAD_DIM_64} and seq_q > {WGPU_BLACKBOX_MAX_VERIFIED_SEQ_Q_FOR_LONG_K_HEAD_DIM_64}, or head_dim=128 with seq_q > {WGPU_BLACKBOX_MAX_VERIFIED_SEQ_Q_FOR_LONG_K_HEAD_DIM_128}); use unit or fallback attention"
        ))));
    }

    if !(problem.dims.seq_q as u32)
        .is_multiple_of(blueprint.tiling_scheme.elements_in_stage_seq_q())
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Stage seq_q must divide problem seq_q".to_string(),
        )));
    }

    if !(problem.dims.head_dim as u32).is_multiple_of(blueprint.tiling_scheme.tile_size.head_dim) {
        return Err(AttentionSetupError::InvalidConfig(Box::new(
            "Tile size head dim must divide problem head dim".to_string(),
        )));
    }

    if blueprint.tiling_scheme.partition_size.head_dim * blueprint.tiling_scheme.tile_size.head_dim
        != problem.dims.head_dim as u32
    {
        return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
            "Tiling scheme's total head dim ({}) does not match problem's head dim ({})",
            blueprint.tiling_scheme.partition_size.head_dim
                * blueprint.tiling_scheme.tile_size.head_dim,
            problem.dims.head_dim
        ))));
    }

    Ok(blueprint)
}

fn wgpu_blackbox_long_k_shape_is_unsafe<R: Runtime>(problem: &AttentionProblem) -> bool {
    let runtime_name = std::any::type_name::<R>();
    if !runtime_name.contains("Wgpu") && !runtime_name.contains("cubecl_wgpu") {
        return false;
    }

    let batch_heads = problem.dims.batch.saturating_mul(problem.dims.num_heads);
    let long_k_head_dim_64 = batch_heads >= 16
        && problem.dims.head_dim == 64
        && problem.dims.seq_kv > WGPU_BLACKBOX_MAX_VERIFIED_SEQ_KV_FOR_HEAD_DIM_64;
    let long_k_head_dim_64 = long_k_head_dim_64
        && problem.dims.seq_q > WGPU_BLACKBOX_MAX_VERIFIED_SEQ_Q_FOR_LONG_K_HEAD_DIM_64;
    let long_k_head_dim_128 = problem.dims.head_dim == 128
        && problem.dims.seq_kv > WGPU_BLACKBOX_MAX_VERIFIED_SEQ_KV_FOR_HEAD_DIM_64
        && problem.dims.seq_q > WGPU_BLACKBOX_MAX_VERIFIED_SEQ_Q_FOR_LONG_K_HEAD_DIM_128;

    long_k_head_dim_64 || long_k_head_dim_128
}

fn blackbox_strategy_for_problem(
    problem: &AttentionProblem,
    strategy: BlackboxAcceleratedStrategy,
) -> BlackboxAcceleratedStrategy {
    let requested_strategy = blackbox_strategy_from_env();
    let batch_heads = problem.dims.batch.saturating_mul(problem.dims.num_heads);
    let long_trellis_slat = batch_heads >= 12
        && problem.dims.head_dim == 128
        && problem.dims.val_dim == 128
        && problem.dims.seq_q >= WGPU_BLACKBOX_TRELLIS_SLAT_MIN_LONG_SEQ
        && problem.dims.seq_kv >= WGPU_BLACKBOX_TRELLIS_SLAT_MIN_LONG_SEQ
        && problem.dims.seq_q <= WGPU_BLACKBOX_MAX_VERIFIED_SEQ_Q_FOR_LONG_K_HEAD_DIM_128;

    if long_trellis_slat {
        let verified = BlackboxAcceleratedStrategy {
            num_planes: 4,
            seq_q: 1,
            seq_kv: 1,
        };
        if let Some(requested) = requested_strategy {
            if requested.num_planes > verified.num_planes
                && !blackbox_allow_unverified_strategy_from_env()
            {
                eprintln!(
                    "CubeK attention blackbox: ignoring unverified long-shape strategy {},{},{} for seq_q={} seq_kv={} head_dim={}; using verified {},{},{} (set CUBEK_ATTENTION_ALLOW_UNVERIFIED_STRATEGY=1 for diagnostics)",
                    requested.num_planes,
                    requested.seq_q,
                    requested.seq_kv,
                    problem.dims.seq_q,
                    problem.dims.seq_kv,
                    problem.dims.head_dim,
                    verified.num_planes,
                    verified.seq_q,
                    verified.seq_kv
                );
                return verified;
            }
            return requested;
        }
        return verified;
    }

    let long_trellis_slat_head_dim_64 = batch_heads >= 16
        && problem.dims.head_dim == 64
        && problem.dims.val_dim == 64
        && problem.dims.seq_q >= WGPU_BLACKBOX_TRELLIS_SLAT_MIN_LONG_SEQ
        && problem.dims.seq_kv >= WGPU_BLACKBOX_TRELLIS_SLAT_MIN_LONG_SEQ
        && problem.dims.seq_q <= WGPU_BLACKBOX_MAX_VERIFIED_SEQ_Q_FOR_LONG_K_HEAD_DIM_64;

    if long_trellis_slat_head_dim_64 {
        let verified = BlackboxAcceleratedStrategy {
            num_planes: 2,
            seq_q: 1,
            seq_kv: 1,
        };
        if let Some(requested) = requested_strategy {
            if requested.num_planes > verified.num_planes
                && !blackbox_allow_unverified_strategy_from_env()
            {
                eprintln!(
                    "CubeK attention blackbox: ignoring unverified long-shape strategy {},{},{} for seq_q={} seq_kv={} head_dim={}; using verified {},{},{} (set CUBEK_ATTENTION_ALLOW_UNVERIFIED_STRATEGY=1 for diagnostics)",
                    requested.num_planes,
                    requested.seq_q,
                    requested.seq_kv,
                    problem.dims.seq_q,
                    problem.dims.seq_kv,
                    problem.dims.head_dim,
                    verified.num_planes,
                    verified.seq_q,
                    verified.seq_kv
                );
                return verified;
            }
            return requested;
        }
        return verified;
    }

    if let Some(strategy) = requested_strategy {
        return strategy;
    }

    strategy
}

fn blackbox_strategy_from_env() -> Option<BlackboxAcceleratedStrategy> {
    let value = std::env::var("CUBEK_ATTENTION_BLACKBOX_STRATEGY").ok()?;
    let mut parts = value.split(',').map(str::trim);
    let num_planes = parts.next()?.parse::<u8>().ok()?;
    let seq_q = parts.next()?.parse::<u8>().ok()?;
    let seq_kv = parts.next()?.parse::<u8>().ok()?;
    if parts.next().is_some() || num_planes == 0 || seq_q == 0 || seq_kv == 0 {
        return None;
    }
    Some(BlackboxAcceleratedStrategy {
        num_planes,
        seq_q,
        seq_kv,
    })
}

fn blackbox_allow_unverified_strategy_from_env() -> bool {
    std::env::var("CUBEK_ATTENTION_ALLOW_UNVERIFIED_STRATEGY")
        .map(|value| !matches!(value.as_str(), "" | "0" | "false" | "False" | "FALSE"))
        .unwrap_or(false)
}

fn maybe_log_strategy(
    problem: &AttentionProblem,
    strategy: &BlackboxAcceleratedStrategy,
    tile_size: AttentionTileSize,
    blueprint: &AttentionBlueprint,
) {
    if std::env::var("CUBEK_ATTENTION_DEBUG_STRATEGY").is_err() {
        return;
    }

    eprintln!(
        concat!(
            "cubek-attention: blackbox strategy ",
            "batch={} heads={} q={} kv={} head_dim={} val_dim={} ",
            "planes={} part_q={} part_kv={} tile_q={} tile_kv={} tile_hd={} tile_vd={} ",
            "stage_q={} check_q={} check_kv={}"
        ),
        problem.dims.batch,
        problem.dims.num_heads,
        problem.dims.seq_q,
        problem.dims.seq_kv,
        problem.dims.head_dim,
        problem.dims.val_dim,
        strategy.num_planes,
        strategy.seq_q,
        strategy.seq_kv,
        tile_size.seq_q,
        tile_size.seq_kv,
        tile_size.head_dim,
        tile_size.val_dim,
        blueprint.tiling_scheme.elements_in_stage_seq_q(),
        blueprint.check_bounds.seq_q,
        blueprint.check_bounds.seq_kv,
    );
}

#[cfg(test)]
mod tests {
    use cubecl::prelude::{AddressType, CubePrimitive};

    use super::*;
    use crate::definition::{
        AccumulatorPrecision, AttentionDims, AttentionGlobalTypes, AttentionOptions,
    };

    fn f16_problem(
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
        val_dim: usize,
    ) -> AttentionProblem {
        AttentionProblem {
            dims: AttentionDims {
                batch: 2,
                num_heads: 12,
                seq_q,
                seq_kv,
                head_dim,
                val_dim,
            },
            masked: false,
            global_dtypes: AttentionGlobalTypes::from_single_float_dtype(
                half::f16::as_type_native_unchecked(),
                u32::as_type_native_unchecked().storage_type(),
            ),
            options: AttentionOptions {
                causal: false,
                accumulator_precision: AccumulatorPrecision::default(),
            },
            address_type: AddressType::default(),
        }
    }

    #[test]
    fn trellis_slat_head_dim_128_uses_verified_strategy_at_8k_plus_tokens() {
        let requested = BlackboxAcceleratedStrategy {
            num_planes: 8,
            seq_q: 1,
            seq_kv: 1,
        };
        let selected =
            blackbox_strategy_for_problem(&f16_problem(10_717, 10_717, 128, 128), requested);
        assert_eq!(selected.num_planes, 4);
        assert_eq!(selected.seq_q, 1);
        assert_eq!(selected.seq_kv, 1);
    }

    #[test]
    fn trellis_slat_head_dim_64_uses_verified_strategy_at_8k_plus_tokens() {
        let requested = BlackboxAcceleratedStrategy {
            num_planes: 8,
            seq_q: 1,
            seq_kv: 1,
        };
        let selected =
            blackbox_strategy_for_problem(&f16_problem(10_717, 10_717, 64, 64), requested);
        assert_eq!(selected.num_planes, 2);
        assert_eq!(selected.seq_q, 1);
        assert_eq!(selected.seq_kv, 1);
    }

    #[test]
    fn short_shapes_keep_inferred_blackbox_strategy() {
        let requested = BlackboxAcceleratedStrategy {
            num_planes: 8,
            seq_q: 1,
            seq_kv: 1,
        };
        let selected =
            blackbox_strategy_for_problem(&f16_problem(4_096, 4_096, 128, 128), requested);
        assert_eq!(selected.num_planes, 8);
        assert_eq!(selected.seq_q, 1);
        assert_eq!(selected.seq_kv, 1);
    }
}
