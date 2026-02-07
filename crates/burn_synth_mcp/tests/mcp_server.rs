use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

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
    client.shutdown();
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
    let output = dir.join("output_mesh.obj");
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
    let mesh_text = fs::read_to_string(&output).expect("failed to read output mesh");
    assert!(
        mesh_text.contains("\nv ") || mesh_text.starts_with("v "),
        "expected OBJ vertex lines"
    );
    let structured = response["result"]["structuredContent"].clone();
    assert_eq!(structured["rmbg_model"], json!("rmbg14"));
    assert_eq!(structured["backend"], json!("cpu"));
    assert_eq!(structured["synthesis_models"], json!(["trellis"]));
    assert_eq!(structured["dry_run"], json!(true));

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
