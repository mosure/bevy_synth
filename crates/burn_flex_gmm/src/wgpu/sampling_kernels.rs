use super::*;

#[cube(launch_unchecked)]
pub(super) fn dense_trilinear_sample_attrs_kernel(
    positions: &Array<f32>,
    occupancy: &Array<i32>,
    attrs: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    dim_x: &f32,
    dim_y: &f32,
    dim_z: &f32,
    max_x: &i32,
    max_y: &i32,
    max_z: &i32,
    stride_x: &i32,
    stride_xy: &i32,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }

    let row = ABSOLUTE_POS;
    let pos_base = row * 3;
    let mut cx = (positions[pos_base] + 0.5) * *dim_x;
    let mut cy = (positions[pos_base + 1] + 0.5) * *dim_y;
    let mut cz = (positions[pos_base + 2] + 0.5) * *dim_z;
    cx = cx.max(0.0).min(*dim_x);
    cy = cy.max(0.0).min(*dim_y);
    cz = cz.max(0.0).min(*dim_z);

    let mut x0 = i32::cast_from((cx - 0.5).max(0.0));
    let mut y0 = i32::cast_from((cy - 0.5).max(0.0));
    let mut z0 = i32::cast_from((cz - 0.5).max(0.0));
    if cx < 0.5 {
        x0 = -1;
    }
    if cy < 0.5 {
        y0 = -1;
    }
    if cz < 0.5 {
        z0 = -1;
    }
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let z1 = z0 + 1;

    let dx0 = cx - f32::cast_from(x0) - 0.5;
    let dy0 = cy - f32::cast_from(y0) - 0.5;
    let dz0 = cz - f32::cast_from(z0) - 0.5;
    let dx1 = cx - f32::cast_from(x1) - 0.5;
    let dy1 = cy - f32::cast_from(y1) - 0.5;
    let dz1 = cz - f32::cast_from(z1) - 0.5;
    let wx0 = (1.0 - dx0.max(-dx0)).max(0.0);
    let wy0 = (1.0 - dy0.max(-dy0)).max(0.0);
    let wz0 = (1.0 - dz0.max(-dz0)).max(0.0);
    let wx1 = (1.0 - dx1.max(-dx1)).max(0.0);
    let wy1 = (1.0 - dy1.max(-dy1)).max(0.0);
    let wz1 = (1.0 - dz1.max(-dz1)).max(0.0);

    let mut a0 = 0.0;
    let mut a1 = 0.0;
    let mut a2 = 0.0;
    let mut a3 = 0.0;
    let mut a4 = 0.0;
    let mut a5 = 0.0;
    let mut wsum = 0.0;

    let w000 = wx0 * wy0 * wz0;
    if w000 > 0.0 && x0 >= 0 && x0 <= *max_x && y0 >= 0 && y0 <= *max_y && z0 >= 0 && z0 <= *max_z {
        let idx000 = usize::cast_from((z0 * *stride_xy + y0 * *stride_x + x0).max(0));
        if occupancy[idx000] > 0 {
            let base = idx000 * 6;
            a0 += attrs[base] * w000;
            a1 += attrs[base + 1] * w000;
            a2 += attrs[base + 2] * w000;
            a3 += attrs[base + 3] * w000;
            a4 += attrs[base + 4] * w000;
            a5 += attrs[base + 5] * w000;
            wsum += w000;
        }
    }
    let w100 = wx1 * wy0 * wz0;
    if w100 > 0.0 && x1 >= 0 && x1 <= *max_x && y0 >= 0 && y0 <= *max_y && z0 >= 0 && z0 <= *max_z {
        let idx100 = usize::cast_from((z0 * *stride_xy + y0 * *stride_x + x1).max(0));
        if occupancy[idx100] > 0 {
            let base = idx100 * 6;
            a0 += attrs[base] * w100;
            a1 += attrs[base + 1] * w100;
            a2 += attrs[base + 2] * w100;
            a3 += attrs[base + 3] * w100;
            a4 += attrs[base + 4] * w100;
            a5 += attrs[base + 5] * w100;
            wsum += w100;
        }
    }
    let w010 = wx0 * wy1 * wz0;
    if w010 > 0.0 && x0 >= 0 && x0 <= *max_x && y1 >= 0 && y1 <= *max_y && z0 >= 0 && z0 <= *max_z {
        let idx010 = usize::cast_from((z0 * *stride_xy + y1 * *stride_x + x0).max(0));
        if occupancy[idx010] > 0 {
            let base = idx010 * 6;
            a0 += attrs[base] * w010;
            a1 += attrs[base + 1] * w010;
            a2 += attrs[base + 2] * w010;
            a3 += attrs[base + 3] * w010;
            a4 += attrs[base + 4] * w010;
            a5 += attrs[base + 5] * w010;
            wsum += w010;
        }
    }
    let w110 = wx1 * wy1 * wz0;
    if w110 > 0.0 && x1 >= 0 && x1 <= *max_x && y1 >= 0 && y1 <= *max_y && z0 >= 0 && z0 <= *max_z {
        let idx110 = usize::cast_from((z0 * *stride_xy + y1 * *stride_x + x1).max(0));
        if occupancy[idx110] > 0 {
            let base = idx110 * 6;
            a0 += attrs[base] * w110;
            a1 += attrs[base + 1] * w110;
            a2 += attrs[base + 2] * w110;
            a3 += attrs[base + 3] * w110;
            a4 += attrs[base + 4] * w110;
            a5 += attrs[base + 5] * w110;
            wsum += w110;
        }
    }
    let w001 = wx0 * wy0 * wz1;
    if w001 > 0.0 && x0 >= 0 && x0 <= *max_x && y0 >= 0 && y0 <= *max_y && z1 >= 0 && z1 <= *max_z {
        let idx001 = usize::cast_from((z1 * *stride_xy + y0 * *stride_x + x0).max(0));
        if occupancy[idx001] > 0 {
            let base = idx001 * 6;
            a0 += attrs[base] * w001;
            a1 += attrs[base + 1] * w001;
            a2 += attrs[base + 2] * w001;
            a3 += attrs[base + 3] * w001;
            a4 += attrs[base + 4] * w001;
            a5 += attrs[base + 5] * w001;
            wsum += w001;
        }
    }
    let w101 = wx1 * wy0 * wz1;
    if w101 > 0.0 && x1 >= 0 && x1 <= *max_x && y0 >= 0 && y0 <= *max_y && z1 >= 0 && z1 <= *max_z {
        let idx101 = usize::cast_from((z1 * *stride_xy + y0 * *stride_x + x1).max(0));
        if occupancy[idx101] > 0 {
            let base = idx101 * 6;
            a0 += attrs[base] * w101;
            a1 += attrs[base + 1] * w101;
            a2 += attrs[base + 2] * w101;
            a3 += attrs[base + 3] * w101;
            a4 += attrs[base + 4] * w101;
            a5 += attrs[base + 5] * w101;
            wsum += w101;
        }
    }
    let w011 = wx0 * wy1 * wz1;
    if w011 > 0.0 && x0 >= 0 && x0 <= *max_x && y1 >= 0 && y1 <= *max_y && z1 >= 0 && z1 <= *max_z {
        let idx011 = usize::cast_from((z1 * *stride_xy + y1 * *stride_x + x0).max(0));
        if occupancy[idx011] > 0 {
            let base = idx011 * 6;
            a0 += attrs[base] * w011;
            a1 += attrs[base + 1] * w011;
            a2 += attrs[base + 2] * w011;
            a3 += attrs[base + 3] * w011;
            a4 += attrs[base + 4] * w011;
            a5 += attrs[base + 5] * w011;
            wsum += w011;
        }
    }
    let w111 = wx1 * wy1 * wz1;
    if w111 > 0.0 && x1 >= 0 && x1 <= *max_x && y1 >= 0 && y1 <= *max_y && z1 >= 0 && z1 <= *max_z {
        let idx111 = usize::cast_from((z1 * *stride_xy + y1 * *stride_x + x1).max(0));
        if occupancy[idx111] > 0 {
            let base = idx111 * 6;
            a0 += attrs[base] * w111;
            a1 += attrs[base + 1] * w111;
            a2 += attrs[base + 2] * w111;
            a3 += attrs[base + 3] * w111;
            a4 += attrs[base + 4] * w111;
            a5 += attrs[base + 5] * w111;
            wsum += w111;
        }
    }

    let out_base = row * 7;
    if wsum > 1.0e-8 {
        let inv = 1.0 / wsum;
        output[out_base] = a0 * inv;
        output[out_base + 1] = a1 * inv;
        output[out_base + 2] = a2 * inv;
        output[out_base + 3] = a3 * inv;
        output[out_base + 4] = a4 * inv;
        output[out_base + 5] = a5 * inv;
        output[out_base + 6] = wsum;
    } else {
        output[out_base] = 0.0;
        output[out_base + 1] = 0.0;
        output[out_base + 2] = 0.0;
        output[out_base + 3] = 0.0;
        output[out_base + 4] = 0.0;
        output[out_base + 5] = 0.0;
        output[out_base + 6] = 0.0;
    }
}
