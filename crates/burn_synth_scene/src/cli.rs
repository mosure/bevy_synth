use std::fs;
use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand};
use serde_json::{Value, json};

use crate::bsn::default_run_id;
use crate::*;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "burn_synth_scene",
    version,
    about = "Scene-image to object-asset composition pipeline"
)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// Plan and generate object images through the OpenAI provider.
    Build(BuildArgs),
    /// Validate a restricted BSN scene file against asset bindings.
    ValidateBsn(ValidateBsnArgs),
    /// Write a Bevy/MCP scene command envelope from a restricted BSN scene.
    WriteCommands(WriteCommandsArgs),
}

#[derive(Parser, Debug, Clone)]
struct BuildArgs {
    #[arg(long)]
    scene: PathBuf,
    #[arg(long, default_value = "docs/input_chair.jpg")]
    object_reference: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 3)]
    candidates: usize,
    #[arg(long, value_enum, default_value_t = SceneQualityProfile::Quality)]
    profile: SceneQualityProfile,
    #[arg(long, default_value = "gpt-5.5")]
    reasoning_model: String,
    #[arg(long, default_value = "gpt-image-2")]
    image_model: String,
}

#[derive(Parser, Debug, Clone)]
struct ValidateBsnArgs {
    #[arg(long)]
    bsn: PathBuf,
    #[arg(long)]
    assets_json: PathBuf,
}

#[derive(Parser, Debug, Clone)]
struct WriteCommandsArgs {
    #[arg(long)]
    bsn: PathBuf,
    #[arg(long)]
    assets_json: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    clear_existing: bool,
    #[arg(long)]
    session_id: Option<String>,
    #[arg(long)]
    sequence: Option<u64>,
}

pub fn run_cli(cli: Cli) -> SceneResult<()> {
    match cli.command {
        Command::Build(args) => {
            let output_dir = args.output_dir.unwrap_or_else(|| {
                PathBuf::from("tmp/runs").join(default_run_id("scene_openai_build"))
            });
            let config = SceneBuildConfig {
                source_scene_path: args.scene,
                object_reference_image_path: args.object_reference,
                output_dir,
                candidate_count: args.candidates,
                quality_profile: args.profile,
                reasoning_model: args.reasoning_model,
                image_model: args.image_model,
                allow_catalog_reuse: false,
            };
            let provider = OpenAiSceneProvider::from_env(OpenAiProviderConfig {
                reasoning_model: config.reasoning_model.clone(),
                image_model: config.image_model.clone(),
                ..OpenAiProviderConfig::default()
            })?;
            let mut pipeline = ScenePipeline::new(config, provider);
            let preparation = pipeline.prepare_openai_inputs()?;
            let output_dir = PathBuf::from(&preparation.output_dir);
            write_json_file(&output_dir.join("preparation.json"), &preparation)?;
            let manifest = pipeline.plan_objects()?;
            write_json_file(&output_dir.join("manifest.json"), &manifest)?;
            let requests = pipeline.prepare_object_image_requests(&manifest)?;
            write_json_file(&output_dir.join("object_image_requests.json"), &requests)?;
            let candidates = pipeline.generate_object_candidates(&requests)?;
            write_json_file(&output_dir.join("object_candidates.json"), &candidates)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "run_id": preparation.run_id,
                    "output_dir": preparation.output_dir,
                    "objects": manifest.objects.len(),
                    "candidates": candidates.len(),
                    "next_stage": "Use burn_synth_mcp images_to_assets with selected object candidate images, then scene_apply_bsn."
                }))
                .unwrap()
            );
            Ok(())
        }
        Command::ValidateBsn(args) => {
            let bsn = fs::read_to_string(args.bsn)?;
            let assets = load_scene_asset_bindings(&args.assets_json)?;
            let parsed = parse_scene_bsn(&bsn, &assets)?;
            println!("{}", serde_json::to_string_pretty(&parsed).unwrap());
            Ok(())
        }
        Command::WriteCommands(args) => {
            let envelope = scene_bsn_file_to_mcp_command_envelope(
                &args.bsn,
                &args.assets_json,
                args.clear_existing,
                args.session_id.as_deref(),
                args.sequence,
            )?;
            write_json_file(&args.output, &envelope)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "output": args.output,
                    "commands": envelope
                        .get("commands")
                        .and_then(Value::as_array)
                        .map(Vec::len)
                        .unwrap_or_default(),
                    "session_id": envelope.get("session_id"),
                    "sequence": envelope.get("sequence"),
                }))
                .unwrap()
            );
            Ok(())
        }
    }
}
