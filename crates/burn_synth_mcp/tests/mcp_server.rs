use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

struct McpTestClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl McpTestClient {
    fn spawn(extra_args: &[&str]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_burn_synth_mcp"));
        command
            .args(extra_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = command.spawn().expect("failed to spawn burn_synth_mcp");
        let stdin = child.stdin.take().expect("missing child stdin");
        let stdout = child.stdout.take().expect("missing child stdout");
        let mut client = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        };
        client.initialize();
        client
    }

    fn initialize(&mut self) {
        let response = self.send_request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "burn_synth_mcp_test", "version": "0.1.0" }
            }),
        );
        assert!(
            response.get("result").is_some(),
            "initialize must return result: {response:#}"
        );
        self.send_notification("notifications/initialized", json!({}));
    }

    fn send_request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        write_framed_json(&mut self.stdin, &request).expect("failed to write request");
        let response = read_framed_json(&mut self.stdout)
            .expect("failed to read response")
            .expect("expected response payload");
        assert_eq!(response["id"], json!(id), "mismatched response id");
        response
    }

    fn send_notification(&mut self, method: &str, params: Value) {
        let request = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        write_framed_json(&mut self.stdin, &request).expect("failed to write notification");
    }

    fn shutdown(mut self) {
        let _ = self.send_request("shutdown", Value::Null);
        self.send_notification("exit", Value::Null);
        let _ = self.child.wait();
    }
}

impl Drop for McpTestClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn initialize_and_list_tools() {
    let mut client = McpTestClient::spawn(&[]);
    let response = client.send_request("tools/list", json!({}));
    let tools = response["result"]["tools"]
        .as_array()
        .expect("tools/list should return array");
    let names = tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert!(names.contains(&"image_to_foreground"));
    assert!(names.contains(&"image_to_mesh"));
    assert!(names.contains(&"images_to_assets"));
    assert!(names.contains(&"scene_clear"));
    assert!(names.contains(&"scene_prepare_build"));
    assert!(names.contains(&"scene_plan_objects"));
    assert!(names.contains(&"scene_generate_object_images"));
    assert!(names.contains(&"scene_build_from_image"));
    assert!(names.contains(&"scene_plan_bsn"));
    assert!(names.contains(&"scene_apply_bsn"));
    assert!(names.contains(&"scene_compose_assets"));
    assert!(names.contains(&"scene_validate_layout"));
    client.shutdown();
}

#[test]
fn scene_apply_bsn_validates_generated_asset_bindings() {
    let mut client = McpTestClient::spawn(&[]);
    let response = client.send_request(
        "tools/call",
        json!({
            "name": "scene_apply_bsn",
            "arguments": {
                "bsn": "synth_scene_v1 {\nasset chair_asset = \"generated:chair_asset\";\nspawn chair_left uses chair_asset translation [-1.0,0.0,2.0] rotation_y 25.0 scale [1.0,1.0,1.0];\n}",
                "asset_bindings": [
                    {
                        "asset_id": "chair_asset",
                        "object_id": "chair_group",
                        "label": "chair",
                        "aliases": ["conference chair"],
                        "path": "/tmp/chair.glb",
                        "reusable": true,
                        "source_image_path": "/tmp/chair.png",
                        "pipeline": "trellis"
                    }
                ],
                "clear_existing": true,
                "apply": false
            }
        }),
    );
    assert!(
        response["result"]["isError"].is_null(),
        "unexpected bsn tool error: {response:#}"
    );
    let structured = &response["result"]["structuredContent"];
    assert_eq!(
        structured["plan"]["placements"].as_array().unwrap().len(),
        1
    );
    assert_eq!(structured["commands"][0]["type"], "clear_scene");
    assert_eq!(structured["commands"][1]["type"], "spawn_path");
    assert_eq!(structured["commands"][1]["cache_key"], "chair_asset");
    client.shutdown();
}

#[test]
fn scene_compose_assets_tool_returns_semantic_layout_plan() {
    let mut client = McpTestClient::spawn(&[]);
    let response = client.send_request(
        "tools/call",
        json!({
            "name": "scene_compose_assets",
            "arguments": {
                "reference_objects": [
                    {
                        "id": "chair_left",
                        "label": "chair",
                        "bbox": [0.10, 0.30, 0.30, 0.85]
                    },
                    {
                        "id": "table_center",
                        "label": "table",
                        "bbox": [0.45, 0.35, 0.78, 0.80]
                    }
                ],
                "assets": [
                    { "reference_id": "chair_left", "path": "/tmp/chair.glb", "label": "chair" },
                    { "reference_id": "table_center", "path": "/tmp/table.glb", "label": "table" }
                ],
                "layout_width": 6.0,
                "layout_depth": 4.0
            }
        }),
    );
    assert!(
        response["result"]["isError"].is_null(),
        "unexpected tool error: {response:#}"
    );
    let structured = &response["result"]["structuredContent"];
    assert_eq!(structured["placements"].as_array().unwrap().len(), 2);
    assert!(
        structured["placements"][0]["translation"][0]
            .as_f64()
            .unwrap()
            < structured["placements"][1]["translation"][0]
                .as_f64()
                .unwrap()
    );
    assert_eq!(
        structured["placements"][0]["cache_key"],
        "path:/tmp/chair.glb"
    );
    client.shutdown();
}

#[test]
fn scene_validate_layout_tool_checks_semantic_and_spatial_match() {
    let mut client = McpTestClient::spawn(&[]);
    let response = client.send_request(
        "tools/call",
        json!({
            "name": "scene_validate_layout",
            "arguments": {
                "reference_objects": [
                    {
                        "id": "chair_left",
                        "label": "chair",
                        "aliases": ["wood chair"],
                        "bbox": [0.10, 0.30, 0.30, 0.85]
                    },
                    {
                        "id": "table_center",
                        "label": "table",
                        "aliases": ["dining table"],
                        "bbox": [0.45, 0.35, 0.78, 0.80]
                    }
                ],
                "scene_status": {
                    "cache_entries": [
                        { "cache_key": "chair_cache", "label": "wood chair", "source_image_path": "chair.png" },
                        { "cache_key": "table_cache", "label": "dining table", "source_image_path": "table.png" }
                    ],
                    "world_items": [
                        { "cache_key": "chair_cache", "translation": [-2.0, 0.0, 0.6], "scale": [1.2, 1.2, 1.2] },
                        { "cache_key": "table_cache", "translation": [1.6, 0.0, 0.5], "scale": [1.8, 1.8, 1.8] }
                    ]
                }
            }
        }),
    );
    assert!(
        response["result"]["isError"].is_null(),
        "unexpected tool error: {response:#}"
    );
    let structured = &response["result"]["structuredContent"];
    assert_eq!(structured["passed"], true, "{structured:#}");
    assert!(structured["scores"]["semantic"].as_f64().unwrap() > 0.9);
    assert!(structured["scores"]["layout"].as_f64().unwrap() >= 0.70);
    client.shutdown();
}

#[test]
fn batch_compose_capture_and_validate_scene_round_trip() {
    let dir = make_temp_dir("scene_round_trip");
    let command_path = dir.join("scene_commands.json");
    let status_path = dir.join("scene.status.json");
    let args = [
        "--backend".to_string(),
        "cpu".to_string(),
        "--scene-control-path".to_string(),
        command_path.display().to_string(),
        "--scene-status-path".to_string(),
        status_path.display().to_string(),
        "--scene-timeout-ms".to_string(),
        "3000".to_string(),
    ];
    let arg_refs = args.iter().map(String::as_str).collect::<Vec<_>>();
    let mut client = McpTestClient::spawn(&arg_refs);

    let chair = dir.join("chair.png");
    let table = dir.join("table.png");
    write_test_image(&chair);
    write_test_image(&table);
    let output_dir = dir.join("assets");
    let batch = client.send_request(
        "tools/call",
        json!({
            "name": "images_to_assets",
            "arguments": {
                "input_image_paths": [
                    chair.display().to_string(),
                    table.display().to_string()
                ],
                "output_dir": output_dir.display().to_string(),
                "synthesis_models": ["triposg"],
                "backend": "cpu",
                "batch_size": 2,
                "dry_run": true
            }
        }),
    );
    assert!(
        batch["result"]["isError"].is_null(),
        "unexpected batch tool error: {batch:#}"
    );
    let batch_content = &batch["result"]["structuredContent"];
    assert_eq!(batch_content["items"].as_array().unwrap().len(), 2);
    assert_eq!(batch_content["stats"]["chunk_size"], json!(2));
    let chair_asset = batch_content["items"][0]["output_path"]
        .as_str()
        .expect("chair output path");
    let table_asset = batch_content["items"][1]["output_path"]
        .as_str()
        .expect("table output path");
    assert!(PathBuf::from(chair_asset).exists());
    assert!(PathBuf::from(table_asset).exists());

    let bridge = start_fake_scene_bridge(command_path.clone(), status_path.clone(), 2);
    let reference_objects = json!([
        {
            "id": "chair_left",
            "label": "chair",
            "aliases": ["chair mesh"],
            "bbox": [0.10, 0.30, 0.30, 0.85]
        },
        {
            "id": "table_center",
            "label": "table",
            "aliases": ["table mesh"],
            "bbox": [0.45, 0.35, 0.78, 0.80]
        }
    ]);
    let compose = client.send_request(
        "tools/call",
        json!({
            "name": "scene_compose_assets",
            "arguments": {
                "reference_objects": reference_objects,
                "assets": [
                    {
                        "reference_id": "chair_left",
                        "label": "chair",
                        "path": chair_asset
                    },
                    {
                        "reference_id": "table_center",
                        "label": "table",
                        "path": table_asset
                    }
                ],
                "layout_width": 6.0,
                "layout_depth": 4.0,
                "clear_existing": true,
                "apply": true
            }
        }),
    );
    assert!(
        compose["result"]["isError"].is_null(),
        "unexpected compose tool error: {compose:#}"
    );
    let compose_content = &compose["result"]["structuredContent"];
    assert_eq!(compose_content["placements"].as_array().unwrap().len(), 2);
    assert_eq!(
        compose_content["acknowledgement"]["acknowledged"],
        json!(true)
    );
    assert_eq!(
        compose_content["acknowledgement"]["status"]["world_items"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    let validate = client.send_request(
        "tools/call",
        json!({
            "name": "scene_validate_layout",
            "arguments": {
                "reference_objects": reference_objects
            }
        }),
    );
    assert!(
        validate["result"]["isError"].is_null(),
        "unexpected validation tool error: {validate:#}"
    );
    let validate_content = &validate["result"]["structuredContent"];
    assert_eq!(validate_content["passed"], true, "{validate_content:#}");
    assert!(validate_content["scores"]["semantic"].as_f64().unwrap() > 0.70);
    assert!(validate_content["scores"]["layout"].as_f64().unwrap() >= 0.70);

    let screenshot = dir.join("capture.png");
    let capture = client.send_request(
        "tools/call",
        json!({
            "name": "scene_capture",
            "arguments": {
                "output_path": screenshot.display().to_string()
            }
        }),
    );
    assert!(
        capture["result"]["isError"].is_null(),
        "unexpected capture tool error: {capture:#}"
    );
    assert!(screenshot.exists(), "expected screenshot artifact");
    assert!(
        capture["result"]["structuredContent"]["acknowledgement"]["status"]["screenshots"]
            .as_array()
            .is_some_and(|screenshots| !screenshots.is_empty())
    );

    let visual_validate = client.send_request(
        "tools/call",
        json!({
            "name": "scene_validate_layout",
            "arguments": {
                "reference_objects": reference_objects,
                "source_image_path": screenshot.display().to_string(),
                "rendered_image_path": screenshot.display().to_string(),
                "thresholds": {
                    "min_image_similarity": 0.99
                }
            }
        }),
    );
    assert!(
        visual_validate["result"]["isError"].is_null(),
        "unexpected visual validation tool error: {visual_validate:#}"
    );
    let visual_content = &visual_validate["result"]["structuredContent"];
    assert_eq!(visual_content["passed"], true, "{visual_content:#}");
    assert!(
        visual_content["image_similarity"]["score"]
            .as_f64()
            .expect("image similarity score")
            >= 0.99
    );

    client.shutdown();
    bridge.join().expect("fake scene bridge should exit");
}

#[test]
fn foreground_tool_dry_run_writes_output() {
    let mut client = McpTestClient::spawn(&[]);
    let dir = make_temp_dir("foreground_dry_run");
    let input = dir.join("input.png");
    let output = dir.join("output_foreground.png");
    write_test_image(&input);

    let response = client.send_request(
        "tools/call",
        json!({
            "name": "image_to_foreground",
            "arguments": {
                "input_image_path": input.display().to_string(),
                "output_image_path": output.display().to_string(),
                "dry_run": true
            }
        }),
    );
    assert!(
        response["result"]["isError"].is_null(),
        "unexpected tool error: {response:#}"
    );
    assert!(
        output.exists(),
        "expected output image {}",
        output.display()
    );
    let structured = response["result"]["structuredContent"].clone();
    assert_eq!(structured["rmbg_model"], json!("rmbg2"));
    assert_eq!(
        structured["output_image_path"],
        json!(output.display().to_string())
    );

    client.shutdown();
}

#[test]
fn foreground_tool_uses_existing_alpha_without_dry_run() {
    let mut client = McpTestClient::spawn(&[]);
    let dir = make_temp_dir("foreground_alpha_passthrough");
    let input = dir.join("input_alpha.png");
    let output = dir.join("output_foreground.png");
    write_alpha_test_image(&input);

    let response = client.send_request(
        "tools/call",
        json!({
            "name": "image_to_foreground",
            "arguments": {
                "input_image_path": input.display().to_string(),
                "output_image_path": output.display().to_string()
            }
        }),
    );
    assert!(
        response["result"]["isError"].is_null(),
        "unexpected tool error: {response:#}"
    );
    assert!(
        output.exists(),
        "expected output image {}",
        output.display()
    );
    let output_image = image::open(&output)
        .expect("failed to open output image")
        .to_rgba8();
    assert_eq!(output_image.get_pixel(0, 0).0[3], 0);
    assert_eq!(output_image.get_pixel(9, 9).0[3], 255);

    client.shutdown();
}

#[test]
fn mesh_tool_uses_cli_model_defaults_in_dry_run() {
    let mut client = McpTestClient::spawn(&[
        "--rmbg-model",
        "rmbg14",
        "--synthesis-models",
        "trellis",
        "--backend",
        "cpu",
    ]);
    let dir = make_temp_dir("mesh_dry_run");
    let input = dir.join("input.png");
    let output = dir.join("output_mesh.glb");
    write_test_image(&input);

    let response = client.send_request(
        "tools/call",
        json!({
            "name": "image_to_mesh",
            "arguments": {
                "input_image_path": input.display().to_string(),
                "output_mesh_path": output.display().to_string(),
                "dry_run": true
            }
        }),
    );
    assert!(
        response["result"]["isError"].is_null(),
        "unexpected tool error: {response:#}"
    );
    assert!(output.exists(), "expected output mesh {}", output.display());
    let mesh_bytes = fs::read(&output).expect("failed to read output mesh");
    assert!(mesh_bytes.starts_with(&[0x67, 0x6C, 0x54, 0x46]));
    let structured = response["result"]["structuredContent"].clone();
    assert_eq!(structured["rmbg_model"], json!("rmbg14"));
    assert_eq!(structured["backend"], json!("cpu"));
    assert_eq!(structured["synthesis_models"], json!(["trellis"]));
    assert_eq!(structured["output_format"], json!("glb"));
    assert_eq!(structured["dry_run"], json!(true));

    client.shutdown();
}

#[test]
fn mesh_tool_supports_target_faces_and_material_metadata_fields() {
    let mut client = McpTestClient::spawn(&[]);
    let dir = make_temp_dir("mesh_target_faces");
    let input = dir.join("input.png");
    let output = dir.join("output_mesh.glb");
    write_test_image(&input);

    let response = client.send_request(
        "tools/call",
        json!({
            "name": "image_to_mesh",
            "arguments": {
                "input_image_path": input.display().to_string(),
                "output_mesh_path": output.display().to_string(),
                "target_faces": 4,
                "dry_run": true
            }
        }),
    );
    assert!(
        response["result"]["isError"].is_null(),
        "unexpected tool error: {response:#}"
    );
    assert!(output.exists(), "expected output mesh {}", output.display());
    let structured = response["result"]["structuredContent"].clone();
    assert_eq!(structured["target_faces"], json!(4));
    assert_eq!(structured["output_format"], json!("glb"));
    assert!(structured["material"].is_null());
    let face_count = structured["faces"]
        .as_u64()
        .expect("faces should be present in structured response");
    assert!(
        face_count <= 4_u64,
        "expected face count <= 4, got {face_count}"
    );

    client.shutdown();
}

#[test]
fn images_to_assets_applies_trellis_pbr_and_target_faces() {
    let mut client = McpTestClient::spawn(&[
        "--synthesis-models",
        "trellis",
        "--backend",
        "cpu",
        "--trellis-pbr",
        "true",
        "--trellis-pbr-texture-size",
        "512",
    ]);
    let dir = make_temp_dir("images_to_assets_trellis_config");
    let input = dir.join("input.png");
    let output_dir = dir.join("assets");
    write_test_image(&input);

    let response = client.send_request(
        "tools/call",
        json!({
            "name": "images_to_assets",
            "arguments": {
                "input_image_paths": [input.display().to_string()],
                "output_dir": output_dir.display().to_string(),
                "target_faces": 42000,
                "trellis_pbr": true,
                "trellis_pbr_texture_size": 1024,
                "dry_run": true
            }
        }),
    );
    assert!(
        response["result"]["isError"].is_null(),
        "unexpected tool error: {response:#}"
    );
    let structured = response["result"]["structuredContent"].clone();
    assert_eq!(structured["synthesis_models"], json!(["trellis"]));
    assert_eq!(structured["backend"], json!("cpu"));
    assert_eq!(structured["target_faces"], json!(42000));
    assert_eq!(structured["trellis_pbr_enabled"], json!(true));
    assert_eq!(structured["trellis_pbr_texture_size"], json!(1024));
    let item = &structured["items"][0];
    assert_eq!(item["target_faces"], json!(42000));
    assert_eq!(item["output_format"], json!("glb"));
    assert!(
        PathBuf::from(item["output_path"].as_str().expect("output path")).exists(),
        "expected dry-run asset output"
    );

    client.shutdown();
}

#[test]
fn images_to_assets_can_promote_outputs_to_shared_catalog() {
    let dir = make_temp_dir("images_to_assets_catalog_promote");
    let catalog_root = dir.join("catalog");
    let mut client = McpTestClient::spawn(&[
        "--synthesis-models",
        "trellis",
        "--backend",
        "cpu",
        "--catalog-cache-root",
        catalog_root.to_str().expect("catalog root path"),
    ]);
    let input = dir.join("chair.png");
    let output_dir = dir.join("assets");
    write_test_image(&input);

    let response = client.send_request(
        "tools/call",
        json!({
            "name": "images_to_assets",
            "arguments": {
                "input_image_paths": [input.display().to_string()],
                "output_dir": output_dir.display().to_string(),
                "synthesis_models": ["trellis"],
                "backend": "cpu",
                "promote_to_catalog": true,
                "dry_run": true
            }
        }),
    );
    assert!(
        response["result"]["isError"].is_null(),
        "unexpected catalog promotion error: {response:#}"
    );
    let structured = &response["result"]["structuredContent"];
    assert_eq!(structured["promote_to_catalog"], json!(true));
    let item = &structured["items"][0];
    let cache_key = item["cache_key"].as_str().expect("cache key");
    assert_eq!(item["catalog_entry"]["cache_key"], json!(cache_key));
    assert_eq!(
        normalize_path_for_compare(
            item["catalog_entry"]["source_image_path"]
                .as_str()
                .expect("catalog source path")
        ),
        normalize_path_for_compare(&input.display().to_string())
    );

    let index_path = catalog_root.join("index.json");
    let index = fs::read_to_string(&index_path).expect("read shared catalog index");
    let index: Value = serde_json::from_str(&index).expect("parse shared catalog index");
    assert_eq!(index["meshes"].as_array().expect("mesh entries").len(), 1);
    assert_eq!(index["meshes"][0]["cache_key"], json!(cache_key));
    let glb_output_id = index["meshes"][0]["glb_output_id"]
        .as_str()
        .expect("glb output id");
    assert!(
        PathBuf::from(glb_output_id).exists(),
        "central cache GLB should exist"
    );

    client.shutdown();
}

fn normalize_path_for_compare(path: &str) -> String {
    path.replace('\\', "/").to_ascii_lowercase()
}

#[test]
fn mesh_tool_rejects_non_glb_output_format() {
    let mut client = McpTestClient::spawn(&[]);
    let dir = make_temp_dir("mesh_glb_only");
    let input = dir.join("input.png");
    write_test_image(&input);

    let rejected_response = client.send_request(
        "tools/call",
        json!({
            "name": "image_to_mesh",
            "arguments": {
                "input_image_path": input.display().to_string(),
                "output_mesh_path": dir.join("output_mesh.obj").display().to_string(),
                "output_format": "gltf",
                "dry_run": true
            }
        }),
    );
    assert!(
        rejected_response["result"]["isError"] == json!(true),
        "expected non-glb output format to be rejected: {rejected_response:#}"
    );

    let glb_output = dir.join("output_mesh.glb");
    let glb_response = client.send_request(
        "tools/call",
        json!({
            "name": "image_to_mesh",
            "arguments": {
                "input_image_path": input.display().to_string(),
                "output_mesh_path": glb_output.display().to_string(),
                "output_format": "glb",
                "dry_run": true
            }
        }),
    );
    assert!(
        glb_response["result"]["isError"].is_null(),
        "unexpected glb tool error: {glb_response:#}"
    );
    assert!(glb_output.exists(), "expected GLB output");
    let glb = fs::read(&glb_output).expect("read glb");
    assert!(glb.starts_with(&[0x67, 0x6C, 0x54, 0x46]));
    assert_eq!(
        glb_response["result"]["structuredContent"]["output_mesh_path"],
        json!(glb_output.display().to_string())
    );
    assert_eq!(
        glb_response["result"]["structuredContent"]["output_format"],
        json!("glb")
    );

    client.shutdown();
}

fn make_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drift")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "burn_synth_mcp_{prefix}_{}_{}",
        std::process::id(),
        nanos
    ));
    fs::create_dir_all(&path).expect("failed to create temp directory");
    path
}

fn write_test_image(path: &PathBuf) {
    let mut image = image::RgbaImage::new(2, 2);
    image.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
    image.put_pixel(1, 0, image::Rgba([0, 255, 0, 255]));
    image.put_pixel(0, 1, image::Rgba([0, 0, 255, 255]));
    image.put_pixel(1, 1, image::Rgba([255, 255, 255, 255]));
    image.save(path).expect("failed to save test image");
}

fn write_alpha_test_image(path: &PathBuf) {
    let mut image = image::RgbaImage::new(10, 10);
    for y in 0..10 {
        for x in 0..10 {
            image.put_pixel(x, y, image::Rgba([200, 160, 120, 255]));
        }
    }
    image.put_pixel(0, 0, image::Rgba([200, 160, 120, 0]));
    image.save(path).expect("failed to save alpha test image");
}

fn start_fake_scene_bridge(
    command_path: PathBuf,
    status_path: PathBuf,
    expected_sequences: usize,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let started = Instant::now();
        let mut processed = 0usize;
        let mut last_sequence = 0u64;
        let mut world_items = Vec::<Value>::new();
        let mut screenshots = Vec::<String>::new();
        while processed < expected_sequences {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "timed out waiting for scene command {}",
                processed + 1
            );
            if !command_path.exists() {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            let command = match fs::read_to_string(&command_path)
                .ok()
                .and_then(|content| serde_json::from_str::<Value>(&content).ok())
            {
                Some(command) => command,
                None => {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                }
            };
            let Some(sequence) = command["sequence"].as_u64() else {
                thread::sleep(Duration::from_millis(10));
                continue;
            };
            if sequence <= last_sequence {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            last_sequence = sequence;
            for scene_command in command["commands"]
                .as_array()
                .expect("scene commands should be an array")
            {
                match scene_command["type"].as_str() {
                    Some("clear_scene") => {
                        world_items.clear();
                    }
                    Some("spawn_path") | Some("spawn_cached") => {
                        let cache_key = scene_command["cache_key"]
                            .as_str()
                            .map(ToOwned::to_owned)
                            .or_else(|| {
                                scene_command["path"]
                                    .as_str()
                                    .map(|path| format!("path:{path}"))
                            })
                            .expect("spawn command should include a cache key or path");
                        world_items.retain(|item| item["cache_key"].as_str() != Some(&cache_key));
                        world_items.push(json!({
                            "cache_key": cache_key,
                            "translation": scene_command["translation"].clone(),
                            "rotation": scene_command["rotation"].clone(),
                            "scale": scene_command["scale"].clone(),
                        }));
                    }
                    Some("capture_screenshot") => {
                        let path = scene_command["path"]
                            .as_str()
                            .expect("capture_screenshot requires path");
                        write_fake_screenshot(&PathBuf::from(path));
                        screenshots.push(path.to_string());
                    }
                    _ => {}
                }
            }
            let status = json!({
                "session_id": command["session_id"].clone(),
                "last_sequence": sequence,
                "ok": true,
                "message": "fake bridge applied",
                "applied_commands": command["commands"].as_array().map_or(0, Vec::len),
                "cache_entries": [],
                "world_items": world_items,
                "camera": null,
                "screenshots": screenshots,
            });
            fs::write(
                &status_path,
                serde_json::to_vec_pretty(&status).expect("serialize fake scene status"),
            )
            .expect("write fake scene status");
            processed += 1;
        }
    })
}

fn write_fake_screenshot(path: &PathBuf) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create screenshot parent");
    }
    let mut image = image::RgbaImage::new(8, 8);
    for y in 0..8 {
        for x in 0..8 {
            let value = if (x + y) % 2 == 0 { 80 } else { 220 };
            image.put_pixel(x, y, image::Rgba([value, value, value, 255]));
        }
    }
    image.save(path).expect("save fake screenshot");
}

fn read_framed_json<R: BufRead + Read>(reader: &mut R) -> io::Result<Option<Value>> {
    let mut content_length = None;
    let mut saw_header = false;
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            if !saw_header {
                return Ok(None);
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF while reading MCP headers",
            ));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        saw_header = true;
        if let Some((name, value)) = trimmed.split_once(':')
            && name.eq_ignore_ascii_case("Content-Length")
        {
            content_length = Some(value.trim().parse::<usize>().map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid Content-Length: {err}"),
                )
            })?);
        }
    }
    let content_length = content_length.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "missing Content-Length header")
    })?;
    let mut payload = vec![0u8; content_length];
    reader.read_exact(&mut payload)?;
    let value = serde_json::from_slice::<Value>(&payload).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid JSON payload: {err}"),
        )
    })?;
    Ok(Some(value))
}

fn write_framed_json<W: Write>(writer: &mut W, value: &Value) -> io::Result<()> {
    let payload = serde_json::to_vec(value).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to serialize JSON payload: {err}"),
        )
    })?;
    write!(writer, "Content-Length: {}\r\n\r\n", payload.len())?;
    writer.write_all(&payload)?;
    writer.flush()
}
