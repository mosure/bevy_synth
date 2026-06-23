use std::borrow::Cow;

// Local adaptation of fslabs/bevy_infinite_grid for the pinned Bevy render API.
use bevy::{
    asset::{load_internal_asset, uuid_handle},
    camera::visibility::{self, NoFrustumCulling, VisibilityClass},
    core_pipeline::core_3d::{Transparent3d, TransparentSortingInfo3d},
    ecs::{
        query::ROQueryItem,
        system::{
            SystemParamItem,
            lifetimeless::{Read, SRes},
        },
    },
    prelude::*,
    render::{
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            BlendState, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
            DepthStencilState, DynamicUniformBuffer, FragmentState, MultisampleState,
            PipelineCache, PolygonMode, PrimitiveState, PrimitiveTopology,
            RenderPipelineDescriptor, ShaderStages, ShaderType, SpecializedRenderPipeline,
            SpecializedRenderPipelines, StencilFaceState, StencilState, TextureFormat, VertexState,
            binding_types::uniform_buffer,
        },
        renderer::{RenderDevice, RenderQueue},
        sync_world::{RenderEntity, SyncToRenderWorld},
        view::{ExtractedView, RenderVisibleEntities},
    },
};

const GRID_SHADER_HANDLE: Handle<Shader> = uuid_handle!("6e7fa981-d772-4a3c-bc46-b45e6a7c10d7");

pub struct InfiniteGridPlugin;

impl Plugin for InfiniteGridPlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        render_app_builder(app);
    }
}

#[derive(Component, Default)]
pub struct InfiniteGrid;

#[derive(Component, Copy, Clone)]
#[require(VisibilityClass)]
#[component(on_add = visibility::add_visibility_class::<InfiniteGridSettings>)]
pub struct InfiniteGridSettings {
    pub x_axis_color: Color,
    pub z_axis_color: Color,
    pub minor_line_color: Color,
    pub major_line_color: Color,
    pub fadeout_distance: f32,
    pub dot_fadeout_strength: f32,
    pub scale: f32,
}

impl Default for InfiniteGridSettings {
    fn default() -> Self {
        Self {
            x_axis_color: Color::srgb(0.92, 0.22, 0.20),
            z_axis_color: Color::srgb(0.20, 0.40, 0.95),
            minor_line_color: Color::srgba(0.34, 0.38, 0.46, 0.24),
            major_line_color: Color::srgba(0.58, 0.64, 0.74, 0.42),
            fadeout_distance: 80.0,
            dot_fadeout_strength: 0.28,
            scale: 1.0,
        }
    }
}

#[derive(Bundle, Default)]
pub struct InfiniteGridBundle {
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub settings: InfiniteGridSettings,
    pub grid: InfiniteGrid,
    pub visibility: Visibility,
    pub view_visibility: ViewVisibility,
    pub inherited_visibility: InheritedVisibility,
    pub shadow_casters: RenderVisibleEntities,
    pub no_frustum_culling: NoFrustumCulling,
    pub sync_to_render_world: SyncToRenderWorld,
}

fn render_app_builder(app: &mut App) {
    load_internal_asset!(
        app,
        GRID_SHADER_HANDLE,
        "infinite_grid.wgsl",
        Shader::from_wgsl
    );

    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<GridViewUniforms>()
        .init_resource::<InfiniteGridUniforms>()
        .init_resource::<GridDisplaySettingsUniforms>()
        .init_resource::<InfiniteGridPipeline>()
        .init_resource::<SpecializedRenderPipelines<InfiniteGridPipeline>>()
        .add_render_command::<Transparent3d, DrawInfiniteGrid>()
        .add_systems(
            ExtractSchedule,
            (extract_infinite_grids, extract_per_camera_settings),
        )
        .add_systems(
            Render,
            (prepare_infinite_grids, prepare_grid_view_uniforms)
                .in_set(RenderSystems::PrepareResources),
        )
        .add_systems(
            Render,
            (
                prepare_bind_groups_for_infinite_grids,
                prepare_grid_view_bind_groups,
            )
                .in_set(RenderSystems::PrepareBindGroups),
        )
        .add_systems(Render, queue_infinite_grids.in_set(RenderSystems::Queue));
}

#[derive(Component)]
struct ExtractedInfiniteGrid {
    transform: GlobalTransform,
    grid: InfiniteGridSettings,
}

#[derive(Debug, ShaderType)]
struct InfiniteGridUniform {
    planar_rotation_matrix: Mat3,
    origin: Vec3,
    normal: Vec3,
}

#[derive(Debug, ShaderType)]
struct GridDisplaySettingsUniform {
    scale: f32,
    dist_fadeout_const: f32,
    dot_fadeout_const: f32,
    x_axis_color: Vec3,
    z_axis_color: Vec3,
    minor_line_color: Vec4,
    major_line_color: Vec4,
}

impl GridDisplaySettingsUniform {
    fn from_settings(settings: &InfiniteGridSettings) -> Self {
        Self {
            scale: settings.scale,
            dist_fadeout_const: 1.0 / settings.fadeout_distance,
            dot_fadeout_const: 1.0 / settings.dot_fadeout_strength,
            x_axis_color: settings.x_axis_color.to_linear().to_vec3(),
            z_axis_color: settings.z_axis_color.to_linear().to_vec3(),
            minor_line_color: settings.minor_line_color.to_linear().to_vec4(),
            major_line_color: settings.major_line_color.to_linear().to_vec4(),
        }
    }
}

#[derive(Resource, Default)]
struct InfiniteGridUniforms {
    uniforms: DynamicUniformBuffer<InfiniteGridUniform>,
}

#[derive(Resource, Default)]
struct GridDisplaySettingsUniforms {
    uniforms: DynamicUniformBuffer<GridDisplaySettingsUniform>,
}

#[derive(Component)]
struct InfiniteGridUniformOffsets {
    position_offset: u32,
    settings_offset: u32,
}

#[derive(Component)]
struct PerCameraSettingsUniformOffset {
    offset: u32,
}

#[derive(Resource)]
struct InfiniteGridBindGroup {
    value: BindGroup,
}

#[derive(Clone, ShaderType)]
struct GridViewUniform {
    projection: Mat4,
    inverse_projection: Mat4,
    view: Mat4,
    inverse_view: Mat4,
    world_position: Vec3,
}

#[derive(Resource, Default)]
struct GridViewUniforms {
    uniforms: DynamicUniformBuffer<GridViewUniform>,
}

#[derive(Component)]
struct GridViewUniformOffset {
    offset: u32,
}

#[derive(Component)]
struct GridViewBindGroup {
    value: BindGroup,
}

struct SetGridViewBindGroup<const I: usize>;

impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetGridViewBindGroup<I> {
    type Param = ();
    type ViewQuery = (Read<GridViewUniformOffset>, Read<GridViewBindGroup>);
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        (view_uniform, bind_group): ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &bind_group.value, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

struct SetInfiniteGridBindGroup<const I: usize>;

impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetInfiniteGridBindGroup<I> {
    type Param = SRes<InfiniteGridBindGroup>;
    type ViewQuery = Option<Read<PerCameraSettingsUniformOffset>>;
    type ItemQuery = Read<InfiniteGridUniformOffsets>;

    fn render<'w>(
        _item: &P,
        camera_settings_offset: ROQueryItem<'w, '_, Self::ViewQuery>,
        base_offsets: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(base_offsets) = base_offsets else {
            warn!("InfiniteGridUniformOffsets missing");
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(
            I,
            &bind_group.into_inner().value,
            &[
                base_offsets.position_offset,
                camera_settings_offset
                    .map(|settings| settings.offset)
                    .unwrap_or(base_offsets.settings_offset),
            ],
        );
        RenderCommandResult::Success
    }
}

struct FinishDrawInfiniteGrid;

impl<P: PhaseItem> RenderCommand<P> for FinishDrawInfiniteGrid {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.draw(0..4, 0..1);
        RenderCommandResult::Success
    }
}

fn prepare_grid_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<GridViewUniforms>,
    views: Query<(Entity, &ExtractedView)>,
) {
    view_uniforms.uniforms.clear();
    for (entity, camera) in views.iter() {
        let projection = camera.clip_from_view;
        let view = camera.world_from_view.to_matrix();
        let inverse_view = view.inverse();
        commands.entity(entity).insert(GridViewUniformOffset {
            offset: view_uniforms.uniforms.push(&GridViewUniform {
                projection,
                inverse_projection: projection.inverse(),
                view,
                inverse_view,
                world_position: camera.world_from_view.translation(),
            }),
        });
    }

    view_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
}

fn prepare_grid_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    uniforms: Res<GridViewUniforms>,
    pipeline: Res<InfiniteGridPipeline>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<Entity, With<GridViewUniformOffset>>,
) {
    let Some(binding) = uniforms.uniforms.binding() else {
        return;
    };

    for entity in views.iter() {
        let bind_group = render_device.create_bind_group(
            "grid-view-bind-group",
            &pipeline_cache.get_bind_group_layout(&pipeline.view_layout),
            &BindGroupEntries::single(binding.clone()),
        );
        commands
            .entity(entity)
            .insert(GridViewBindGroup { value: bind_group });
    }
}

fn extract_infinite_grids(
    mut commands: Commands,
    grids: Extract<
        Query<(
            RenderEntity,
            &InfiniteGridSettings,
            &GlobalTransform,
            &RenderVisibleEntities,
        )>,
    >,
) {
    let extracted = grids
        .iter()
        .map(|(entity, grid, transform, visible_entities)| {
            (
                entity,
                (
                    ExtractedInfiniteGrid {
                        transform: *transform,
                        grid: *grid,
                    },
                    visible_entities.clone(),
                ),
            )
        })
        .collect::<Vec<_>>();
    commands.try_insert_batch(extracted);
}

fn extract_per_camera_settings(
    mut commands: Commands,
    cameras: Extract<Query<(RenderEntity, &InfiniteGridSettings), With<Camera>>>,
) {
    let extracted = cameras
        .iter()
        .map(|(entity, settings)| (entity, *settings))
        .collect::<Vec<_>>();
    commands.try_insert_batch(extracted);
}

fn prepare_infinite_grids(
    mut commands: Commands,
    grids: Query<(Entity, &ExtractedInfiniteGrid)>,
    cameras: Query<(Entity, &InfiniteGridSettings), With<ExtractedView>>,
    mut position_uniforms: ResMut<InfiniteGridUniforms>,
    mut settings_uniforms: ResMut<GridDisplaySettingsUniforms>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    position_uniforms.uniforms.clear();
    settings_uniforms.uniforms.clear();

    for (entity, extracted) in &grids {
        let transform = extracted.transform;
        let t = transform.compute_transform();
        commands.entity(entity).insert(InfiniteGridUniformOffsets {
            position_offset: position_uniforms.uniforms.push(&InfiniteGridUniform {
                planar_rotation_matrix: Mat3::from_quat(t.rotation.inverse()),
                origin: transform.translation(),
                normal: *transform.up(),
            }),
            settings_offset: settings_uniforms
                .uniforms
                .push(&GridDisplaySettingsUniform::from_settings(&extracted.grid)),
        });
    }

    for (entity, settings) in &cameras {
        commands
            .entity(entity)
            .insert(PerCameraSettingsUniformOffset {
                offset: settings_uniforms
                    .uniforms
                    .push(&GridDisplaySettingsUniform::from_settings(settings)),
            });
    }

    position_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
    settings_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
}

fn prepare_bind_groups_for_infinite_grids(
    mut commands: Commands,
    position_uniforms: Res<InfiniteGridUniforms>,
    settings_uniforms: Res<GridDisplaySettingsUniforms>,
    pipeline: Res<InfiniteGridPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    let Some((position_binding, settings_binding)) = position_uniforms
        .uniforms
        .binding()
        .zip(settings_uniforms.uniforms.binding())
    else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "infinite-grid-bind-group",
        &pipeline_cache.get_bind_group_layout(&pipeline.infinite_grid_layout),
        &BindGroupEntries::sequential((position_binding.clone(), settings_binding.clone())),
    );
    commands.insert_resource(InfiniteGridBindGroup { value: bind_group });
}

fn queue_infinite_grids(
    pipeline_cache: Res<PipelineCache>,
    transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
    pipeline: Res<InfiniteGridPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<InfiniteGridPipeline>>,
    infinite_grids: Query<&ExtractedInfiniteGrid>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    mut views: Query<(&ExtractedView, &RenderVisibleEntities, &Msaa)>,
) {
    let Some(draw_function_id) = transparent_draw_functions
        .read()
        .get_id::<DrawInfiniteGrid>()
    else {
        return;
    };

    for (view, entities, msaa) in views.iter_mut() {
        let Some(phase) = transparent_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            GridPipelineKey {
                target_format: view.target_format,
                sample_count: msaa.samples(),
            },
        );
        let Some(visible_grids) = entities
            .classes
            .get(&std::any::TypeId::of::<InfiniteGridSettings>())
        else {
            continue;
        };
        for (render_entity, main_entity) in visible_grids.iter_visible() {
            if !infinite_grids
                .get(*render_entity)
                .map(|grid| plane_check(&grid.transform, view.world_from_view.translation()))
                .unwrap_or(false)
            {
                continue;
            }
            phase.add_retained(Transparent3d {
                sorting_info: TransparentSortingInfo3d::AlwaysOnTop,
                pipeline: pipeline_id,
                entity: (*render_entity, *main_entity),
                draw_function: draw_function_id,
                distance: f32::NEG_INFINITY,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: false,
            });
        }
    }
}

fn plane_check(plane: &GlobalTransform, point: Vec3) -> bool {
    plane.up().dot(plane.translation() - point).abs() > f32::EPSILON
}

type DrawInfiniteGrid = (
    SetItemPipeline,
    SetGridViewBindGroup<0>,
    SetInfiniteGridBindGroup<1>,
    FinishDrawInfiniteGrid,
);

#[derive(Resource)]
struct InfiniteGridPipeline {
    view_layout: BindGroupLayoutDescriptor,
    infinite_grid_layout: BindGroupLayoutDescriptor,
}

impl FromWorld for InfiniteGridPipeline {
    fn from_world(_world: &mut World) -> Self {
        let view_layout = BindGroupLayoutDescriptor::new(
            "grid-view-bind-group-layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                uniform_buffer::<GridViewUniform>(true),
            ),
        );
        let infinite_grid_layout = BindGroupLayoutDescriptor::new(
            "infinite-grid-bind-group-layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer::<InfiniteGridUniform>(true),
                    uniform_buffer::<GridDisplaySettingsUniform>(true),
                ),
            ),
        );

        Self {
            view_layout,
            infinite_grid_layout,
        }
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct GridPipelineKey {
    target_format: TextureFormat,
    sample_count: u32,
}

impl SpecializedRenderPipeline for InfiniteGridPipeline {
    type Key = GridPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some(Cow::Borrowed("infinite-grid-render-pipeline")),
            layout: vec![self.view_layout.clone(), self.infinite_grid_layout.clone()],
            immediate_size: 0,
            vertex: VertexState {
                shader: GRID_SHADER_HANDLE,
                shader_defs: Vec::new(),
                entry_point: Some(Cow::Borrowed("vertex")),
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: bevy::render::render_resource::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Greater),
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: GRID_SHADER_HANDLE,
                shader_defs: Vec::new(),
                entry_point: Some(Cow::Borrowed("fragment")),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: false,
        }
    }
}
