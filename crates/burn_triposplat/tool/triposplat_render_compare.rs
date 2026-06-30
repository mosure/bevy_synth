use std::{
    cmp::Ordering,
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use image::{ImageBuffer, Rgba, RgbaImage};
use serde::Serialize;

const SPLAT_RECORD_BYTES: usize = 32;
const BACKGROUND: [f32; 3] = [0.05, 0.055, 0.065];

#[derive(Debug, Parser)]
#[command(about = "Render and compare TripoSplat Gaussian outputs with image-level PSNR.")]
struct Args {
    #[arg(long)]
    reference_splat: Option<PathBuf>,

    #[arg(long)]
    candidate_splat: Option<PathBuf>,

    #[arg(long)]
    reference_image: Option<PathBuf>,

    #[arg(long)]
    candidate_image: Option<PathBuf>,

    #[arg(long)]
    reference_render: Option<PathBuf>,

    #[arg(long)]
    candidate_render: Option<PathBuf>,

    #[arg(long)]
    report: Option<PathBuf>,

    #[arg(long, default_value_t = 256)]
    width: u32,

    #[arg(long, default_value_t = 256)]
    height: u32,

    #[arg(long, default_value_t = 1.35)]
    frame_margin: f32,

    #[arg(long, default_value_t = 1.0)]
    splat_scale: f32,

    #[arg(long, default_value_t = 0.65)]
    min_sigma_px: f32,

    #[arg(long, default_value_t = 16.0)]
    max_sigma_px: f32,

    #[arg(long, default_value_t = 3.0)]
    sigma_extent: f32,

    #[arg(long, default_value_t = 35.0)]
    min_psnr: f64,

    #[arg(long, default_value_t = false)]
    include_alpha: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    reference_splat: Option<String>,
    candidate_splat: Option<String>,
    reference_image: Option<String>,
    candidate_image: Option<String>,
    reference_render: Option<String>,
    candidate_render: Option<String>,
    width: u32,
    height: u32,
    reference_records: Option<usize>,
    candidate_records: Option<usize>,
    render: RenderSettings,
    compare: ImageDiff,
    failures: Vec<String>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct RenderSettings {
    frame_margin: f32,
    splat_scale: f32,
    min_sigma_px: f32,
    max_sigma_px: f32,
    sigma_extent: f32,
    include_alpha: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ImageDiff {
    channels: usize,
    pixels: usize,
    mse: f64,
    psnr: Option<f64>,
    psnr_infinite: bool,
    mean_abs: f64,
    max_abs: u8,
    max_abs_pixel: [u32; 2],
    max_abs_channel: usize,
    reference_at_max_abs: u8,
    candidate_at_max_abs: u8,
}

#[derive(Clone, Copy, Debug)]
struct SplatRecord {
    position: [f32; 3],
    scale: [f32; 3],
    rgba: [u8; 4],
}

#[derive(Clone, Copy, Debug)]
struct Frame {
    center: [f32; 3],
    radius: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    validate_args(&args)?;

    let mut reference_records = None;
    let mut candidate_records = None;

    let reference = if let Some(path) = &args.reference_image {
        image::open(path)?.to_rgba8()
    } else {
        let reference_path = args
            .reference_splat
            .as_ref()
            .expect("validated reference splat");
        let candidate_path = args
            .candidate_splat
            .as_ref()
            .expect("validated candidate splat");
        let reference_splats = read_splat(reference_path)?;
        let candidate_splats = read_splat(candidate_path)?;
        reference_records = Some(reference_splats.len());
        candidate_records = Some(candidate_splats.len());
        let frame = shared_frame(&reference_splats, &candidate_splats, args.frame_margin)?;
        render_splats(&reference_splats, frame, &args)
    };

    let candidate = if let Some(path) = &args.candidate_image {
        image::open(path)?.to_rgba8()
    } else {
        let reference_path = args
            .reference_splat
            .as_ref()
            .expect("validated reference splat");
        let candidate_path = args
            .candidate_splat
            .as_ref()
            .expect("validated candidate splat");
        let reference_splats = read_splat(reference_path)?;
        let candidate_splats = read_splat(candidate_path)?;
        if reference_records.is_none() {
            reference_records = Some(reference_splats.len());
        }
        if candidate_records.is_none() {
            candidate_records = Some(candidate_splats.len());
        }
        let frame = shared_frame(&reference_splats, &candidate_splats, args.frame_margin)?;
        render_splats(&candidate_splats, frame, &args)
    };

    if let Some(path) = &args.reference_render {
        write_image(path, &reference)?;
    }
    if let Some(path) = &args.candidate_render {
        write_image(path, &candidate)?;
    }

    let compare = image_diff(&reference, &candidate, args.include_alpha)?;
    let mut failures = Vec::new();
    let psnr_for_threshold = compare.psnr.unwrap_or(f64::INFINITY);
    if psnr_for_threshold < args.min_psnr {
        failures.push(format!(
            "psnr {:.3} dB < {:.3} dB",
            psnr_for_threshold, args.min_psnr
        ));
    }
    if reference.width() != candidate.width() || reference.height() != candidate.height() {
        failures.push(format!(
            "image size mismatch: reference={}x{} candidate={}x{}",
            reference.width(),
            reference.height(),
            candidate.width(),
            candidate.height()
        ));
    }

    let report = Report {
        reference_splat: args.reference_splat.as_ref().map(display_path),
        candidate_splat: args.candidate_splat.as_ref().map(display_path),
        reference_image: args.reference_image.as_ref().map(display_path),
        candidate_image: args.candidate_image.as_ref().map(display_path),
        reference_render: args.reference_render.as_ref().map(display_path),
        candidate_render: args.candidate_render.as_ref().map(display_path),
        width: reference.width(),
        height: reference.height(),
        reference_records,
        candidate_records,
        render: RenderSettings {
            frame_margin: args.frame_margin,
            splat_scale: args.splat_scale,
            min_sigma_px: args.min_sigma_px,
            max_sigma_px: args.max_sigma_px,
            sigma_extent: args.sigma_extent,
            include_alpha: args.include_alpha,
        },
        compare,
        passed: failures.is_empty(),
        failures,
    };

    let text = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = &args.report {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &text)?;
    }
    print!("{text}");

    if report.passed {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

fn validate_args(args: &Args) -> Result<(), String> {
    let has_splats = args.reference_splat.is_some() || args.candidate_splat.is_some();
    let has_images = args.reference_image.is_some() || args.candidate_image.is_some();
    if has_splats && has_images {
        return Err("use either --reference-splat/--candidate-splat or --reference-image/--candidate-image, not both".to_string());
    }
    if has_splats {
        if args.reference_splat.is_none() || args.candidate_splat.is_none() {
            return Err(
                "splat mode requires both --reference-splat and --candidate-splat".to_string(),
            );
        }
    } else if has_images {
        if args.reference_image.is_none() || args.candidate_image.is_none() {
            return Err(
                "image mode requires both --reference-image and --candidate-image".to_string(),
            );
        }
    } else {
        return Err("provide either splat inputs or image inputs".to_string());
    }
    if args.width == 0 || args.height == 0 {
        return Err("render dimensions must be non-zero".to_string());
    }
    if !args.frame_margin.is_finite() || args.frame_margin <= 0.0 {
        return Err("frame margin must be finite and positive".to_string());
    }
    if !args.splat_scale.is_finite() || args.splat_scale <= 0.0 {
        return Err("splat scale must be finite and positive".to_string());
    }
    if !args.min_sigma_px.is_finite() || args.min_sigma_px <= 0.0 {
        return Err("min sigma must be finite and positive".to_string());
    }
    if !args.max_sigma_px.is_finite() || args.max_sigma_px < args.min_sigma_px {
        return Err("max sigma must be finite and >= min sigma".to_string());
    }
    if !args.sigma_extent.is_finite() || args.sigma_extent <= 0.0 {
        return Err("sigma extent must be finite and positive".to_string());
    }
    Ok(())
}

fn read_splat(path: &PathBuf) -> Result<Vec<SplatRecord>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % SPLAT_RECORD_BYTES != 0 {
        return Err(format!(
            "{} byte length {} is not divisible by {SPLAT_RECORD_BYTES}",
            path.display(),
            bytes.len()
        )
        .into());
    }
    let records = bytes
        .as_chunks::<SPLAT_RECORD_BYTES>()
        .0
        .iter()
        .map(|chunk| SplatRecord {
            position: [
                read_f32_le(chunk, 0),
                read_f32_le(chunk, 4),
                read_f32_le(chunk, 8),
            ],
            scale: [
                read_f32_le(chunk, 12),
                read_f32_le(chunk, 16),
                read_f32_le(chunk, 20),
            ],
            rgba: [chunk[24], chunk[25], chunk[26], chunk[27]],
        })
        .collect::<Vec<_>>();
    if records.is_empty() {
        return Err(format!("{} contains no splat records", path.display()).into());
    }
    Ok(records)
}

fn read_f32_le(chunk: &[u8], start: usize) -> f32 {
    f32::from_le_bytes([
        chunk[start],
        chunk[start + 1],
        chunk[start + 2],
        chunk[start + 3],
    ])
}

fn shared_frame(
    reference: &[SplatRecord],
    candidate: &[SplatRecord],
    margin: f32,
) -> Result<Frame, String> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for splat in reference.iter().chain(candidate.iter()) {
        for axis in 0..3 {
            let extent = splat.scale[axis].abs().max(1.0e-4) * 3.0;
            min[axis] = min[axis].min(splat.position[axis] - extent);
            max[axis] = max[axis].max(splat.position[axis] + extent);
        }
    }
    if min.iter().any(|value| !value.is_finite()) || max.iter().any(|value| !value.is_finite()) {
        return Err("cannot frame non-finite splat bounds".to_string());
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let radius = ((max[0] - min[0]).max(max[1] - min[1]) * 0.5 * margin).max(1.0e-4);
    Ok(Frame { center, radius })
}

fn render_splats(records: &[SplatRecord], frame: Frame, args: &Args) -> RgbaImage {
    let mut accum = vec![
        [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 1.0f32];
        (args.width * args.height) as usize
    ];
    let mut order = (0..records.len()).collect::<Vec<_>>();
    order.sort_by(|a, b| {
        records[*a].position[2]
            .partial_cmp(&records[*b].position[2])
            .unwrap_or(Ordering::Equal)
    });

    let pixel_scale = (args.width.min(args.height) as f32 - 1.0) / (2.0 * frame.radius);
    for index in order {
        let splat = records[index];
        if !valid_splat(&splat) {
            continue;
        }
        let px =
            ((splat.position[0] - frame.center[0]) * pixel_scale + args.width as f32 * 0.5).round();
        let py = (args.height as f32 * 0.5 - (splat.position[1] - frame.center[1]) * pixel_scale)
            .round();
        let sigma =
            (splat.scale[0].abs().max(splat.scale[1].abs()) * pixel_scale * args.splat_scale)
                .clamp(args.min_sigma_px, args.max_sigma_px);
        let radius = (sigma * args.sigma_extent).ceil() as i32;
        let min_x = (px as i32 - radius).max(0);
        let max_x = (px as i32 + radius).min(args.width as i32 - 1);
        let min_y = (py as i32 - radius).max(0);
        let max_y = (py as i32 + radius).min(args.height as i32 - 1);
        let color = [
            splat.rgba[0] as f32 / 255.0,
            splat.rgba[1] as f32 / 255.0,
            splat.rgba[2] as f32 / 255.0,
        ];
        let opacity = splat.rgba[3] as f32 / 255.0;
        let inv_two_sigma_sq = 0.5 / (sigma * sigma);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - px;
                let dy = y as f32 - py;
                let alpha = opacity * (-(dx * dx + dy * dy) * inv_two_sigma_sq).exp();
                if alpha <= 1.0e-4 {
                    continue;
                }
                let dst = &mut accum[(y as u32 * args.width + x as u32) as usize];
                for channel in 0..3 {
                    dst[channel] = color[channel] * alpha + dst[channel] * (1.0 - alpha);
                }
                dst[3] = alpha + dst[3] * (1.0 - alpha);
            }
        }
    }

    let mut image = ImageBuffer::new(args.width, args.height);
    for (idx, pixel) in image.pixels_mut().enumerate() {
        let rgba = accum[idx];
        *pixel = Rgba([
            to_u8(rgba[0] * 255.0),
            to_u8(rgba[1] * 255.0),
            to_u8(rgba[2] * 255.0),
            to_u8(rgba[3] * 255.0),
        ]);
    }
    image
}

fn valid_splat(splat: &SplatRecord) -> bool {
    splat.position.iter().all(|value| value.is_finite())
        && splat
            .scale
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
}

fn to_u8(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

fn write_image(path: &PathBuf, image: &RgbaImage) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    image.save(path)?;
    Ok(())
}

fn image_diff(
    reference: &RgbaImage,
    candidate: &RgbaImage,
    include_alpha: bool,
) -> Result<ImageDiff, String> {
    if reference.dimensions() != candidate.dimensions() {
        return Err(format!(
            "image dimensions differ: reference={:?} candidate={:?}",
            reference.dimensions(),
            candidate.dimensions()
        ));
    }
    let channels = if include_alpha { 4 } else { 3 };
    let mut sum_sq = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0u8;
    let mut max_abs_pixel = [0, 0];
    let mut max_abs_channel = 0usize;
    let mut reference_at_max_abs = 0u8;
    let mut candidate_at_max_abs = 0u8;
    for y in 0..reference.height() {
        for x in 0..reference.width() {
            let ref_pixel = reference.get_pixel(x, y).0;
            let cand_pixel = candidate.get_pixel(x, y).0;
            for channel in 0..channels {
                let diff = ref_pixel[channel].abs_diff(cand_pixel[channel]);
                let diff_f64 = diff as f64;
                sum_sq += diff_f64 * diff_f64;
                sum_abs += diff_f64;
                if diff > max_abs {
                    max_abs = diff;
                    max_abs_pixel = [x, y];
                    max_abs_channel = channel;
                    reference_at_max_abs = ref_pixel[channel];
                    candidate_at_max_abs = cand_pixel[channel];
                }
            }
        }
    }
    let samples = (reference.width() as usize * reference.height() as usize * channels).max(1);
    let mse = sum_sq / samples as f64;
    let (psnr, psnr_infinite) = if mse == 0.0 {
        (None, true)
    } else {
        (Some(20.0 * (255.0 / mse.sqrt()).log10()), false)
    };
    Ok(ImageDiff {
        channels,
        pixels: (reference.width() * reference.height()) as usize,
        mse,
        psnr,
        psnr_infinite,
        mean_abs: sum_abs / samples as f64,
        max_abs,
        max_abs_pixel,
        max_abs_channel,
        reference_at_max_abs,
        candidate_at_max_abs,
    })
}

fn display_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string()
}
