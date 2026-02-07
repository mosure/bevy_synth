use burn::tensor::ops::InterpolateMode;

pub fn resize_chw_align_corners_false(
    input: &[f32],
    channels: usize,
    in_height: usize,
    in_width: usize,
    out_height: usize,
    out_width: usize,
    mode: InterpolateMode,
) -> Vec<f32> {
    if in_height == out_height && in_width == out_width {
        return input.to_vec();
    }

    match mode {
        InterpolateMode::Nearest => resize_nearest_align_corners_false(
            input, channels, in_height, in_width, out_height, out_width,
        ),
        InterpolateMode::Bicubic | InterpolateMode::Bilinear => {
            if out_height < in_height || out_width < in_width {
                resize_area_align_corners_false(
                    input, channels, in_height, in_width, out_height, out_width,
                )
            } else {
                resize_bilinear_align_corners_false(
                    input, channels, in_height, in_width, out_height, out_width,
                )
            }
        }
    }
}

fn resize_nearest_align_corners_false(
    input: &[f32],
    channels: usize,
    in_height: usize,
    in_width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * out_height * out_width];
    let scale_y = in_height as f32 / out_height as f32;
    let scale_x = in_width as f32 / out_width as f32;

    for c in 0..channels {
        let in_base = c * in_height * in_width;
        let out_base = c * out_height * out_width;
        for oy in 0..out_height {
            let in_y = ((oy as f32 + 0.5) * scale_y - 0.5).round();
            let iy = in_y.clamp(0.0, (in_height - 1) as f32).round() as usize;
            for ox in 0..out_width {
                let in_x = ((ox as f32 + 0.5) * scale_x - 0.5).round();
                let ix = in_x.clamp(0.0, (in_width - 1) as f32).round() as usize;
                output[out_base + oy * out_width + ox] = input[in_base + iy * in_width + ix];
            }
        }
    }

    output
}

fn resize_bilinear_align_corners_false(
    input: &[f32],
    channels: usize,
    in_height: usize,
    in_width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * out_height * out_width];
    let scale_y = in_height as f32 / out_height as f32;
    let scale_x = in_width as f32 / out_width as f32;
    let max_y = in_height.saturating_sub(1) as isize;
    let max_x = in_width.saturating_sub(1) as isize;

    for c in 0..channels {
        let in_base = c * in_height * in_width;
        let out_base = c * out_height * out_width;
        for oy in 0..out_height {
            let in_y = (oy as f32 + 0.5) * scale_y - 0.5;
            let y0 = in_y.floor() as isize;
            let y1 = y0 + 1;
            let wy = in_y - y0 as f32;
            let y0c = y0.clamp(0, max_y) as usize;
            let y1c = y1.clamp(0, max_y) as usize;
            for ox in 0..out_width {
                let in_x = (ox as f32 + 0.5) * scale_x - 0.5;
                let x0 = in_x.floor() as isize;
                let x1 = x0 + 1;
                let wx = in_x - x0 as f32;
                let x0c = x0.clamp(0, max_x) as usize;
                let x1c = x1.clamp(0, max_x) as usize;

                let v00 = input[in_base + y0c * in_width + x0c];
                let v01 = input[in_base + y0c * in_width + x1c];
                let v10 = input[in_base + y1c * in_width + x0c];
                let v11 = input[in_base + y1c * in_width + x1c];

                let top = v00 * (1.0 - wx) + v01 * wx;
                let bottom = v10 * (1.0 - wx) + v11 * wx;
                output[out_base + oy * out_width + ox] = top * (1.0 - wy) + bottom * wy;
            }
        }
    }

    output
}

fn resize_area_align_corners_false(
    input: &[f32],
    channels: usize,
    in_height: usize,
    in_width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; channels * out_height * out_width];
    let scale_y = in_height as f32 / out_height as f32;
    let scale_x = in_width as f32 / out_width as f32;
    let in_height_f = in_height as f32;
    let in_width_f = in_width as f32;

    for c in 0..channels {
        let in_base = c * in_height * in_width;
        let out_base = c * out_height * out_width;
        for oy in 0..out_height {
            let y0 = oy as f32 * scale_y;
            let y1 = (oy + 1) as f32 * scale_y;
            let y_start = y0.floor() as isize;
            let y_end = y1.ceil() as isize;
            for ox in 0..out_width {
                let x0 = ox as f32 * scale_x;
                let x1 = (ox + 1) as f32 * scale_x;
                let x_start = x0.floor() as isize;
                let x_end = x1.ceil() as isize;

                let mut acc = 0.0f32;
                let mut total = 0.0f32;
                for iy in y_start..y_end {
                    if iy < 0 || iy >= in_height as isize {
                        continue;
                    }
                    let iy0 = iy as f32;
                    let iy1 = (iy0 + 1.0).min(in_height_f);
                    let overlap_y = (y1.min(iy1) - y0.max(iy0)).max(0.0);
                    if overlap_y <= 0.0 {
                        continue;
                    }
                    for ix in x_start..x_end {
                        if ix < 0 || ix >= in_width as isize {
                            continue;
                        }
                        let ix0 = ix as f32;
                        let ix1 = (ix0 + 1.0).min(in_width_f);
                        let overlap_x = (x1.min(ix1) - x0.max(ix0)).max(0.0);
                        if overlap_x <= 0.0 {
                            continue;
                        }
                        let weight = overlap_x * overlap_y;
                        let idx = in_base + iy as usize * in_width + ix as usize;
                        acc += input[idx] * weight;
                        total += weight;
                    }
                }

                if total > 0.0 {
                    output[out_base + oy * out_width + ox] = acc / total;
                }
            }
        }
    }

    output
}
