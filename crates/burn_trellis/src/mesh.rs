use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn new(vertices: Vec<[f32; 3]>, faces: Vec<[u32; 3]>) -> Self {
        Self { vertices, faces }
    }
}

pub fn load_obj_mesh(path: &Path) -> Result<Mesh, String> {
    let file =
        fs::File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let reader = BufReader::new(file);
    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|err| format!("failed to read OBJ line: {err}"))?;
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("v ") {
            let parts = rest.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 3 {
                continue;
            }
            let x = parts[0]
                .parse::<f32>()
                .map_err(|err| format!("invalid OBJ vertex x '{}': {err}", parts[0]))?;
            let y = parts[1]
                .parse::<f32>()
                .map_err(|err| format!("invalid OBJ vertex y '{}': {err}", parts[1]))?;
            let z = parts[2]
                .parse::<f32>()
                .map_err(|err| format!("invalid OBJ vertex z '{}': {err}", parts[2]))?;
            vertices.push([x, y, z]);
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("f ") {
            let parts = rest.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 3 {
                continue;
            }
            let mut idx = [0u32; 3];
            for i in 0..3 {
                let value = parts[i]
                    .split('/')
                    .next()
                    .ok_or_else(|| format!("invalid OBJ face index '{}'", parts[i]))?;
                let parsed = value
                    .parse::<u32>()
                    .map_err(|err| format!("invalid OBJ face index '{}': {err}", value))?;
                idx[i] = parsed.saturating_sub(1);
            }
            faces.push(idx);
        }
    }

    if vertices.is_empty() || faces.is_empty() {
        return Err(format!(
            "OBJ '{}' did not contain vertices/faces",
            path.display()
        ));
    }
    Ok(Mesh { vertices, faces })
}

pub fn write_obj_mesh(path: &Path, mesh: &Mesh) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }
    let file = fs::File::create(path)
        .map_err(|err| format!("failed to create '{}': {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for vertex in &mesh.vertices {
        writeln!(writer, "v {} {} {}", vertex[0], vertex[1], vertex[2])
            .map_err(|err| format!("failed to write vertex: {err}"))?;
    }
    for face in &mesh.faces {
        writeln!(writer, "f {} {} {}", face[0] + 1, face[1] + 1, face[2] + 1)
            .map_err(|err| format!("failed to write face: {err}"))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush OBJ: {err}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{Mesh, load_obj_mesh, write_obj_mesh};

    #[test]
    fn obj_roundtrip_works() {
        let mesh = Mesh {
            vertices: vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            faces: vec![[0, 1, 2]],
        };
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("burn_trellis_mesh_{unique}.obj"));
        write_obj_mesh(&path, &mesh).expect("failed to write obj");
        let loaded = load_obj_mesh(&path).expect("failed to read obj");
        assert_eq!(loaded, mesh);
        let _ = std::fs::remove_file(path);
    }
}
