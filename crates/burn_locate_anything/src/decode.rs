use serde::{Deserialize, Serialize};

use crate::LocateAnythingModelConfig;
use crate::{
    Detection, LocateAnythingError, LocateAnythingResult, normalize_bbox, normalize_point,
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeMode {
    ParallelBox,
    Autoregressive,
    #[default]
    Hybrid,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelPatternKind {
    CoordBox,
    EmptyBox,
    ErrorBox,
    ImEnd,
    PointBox,
    RefObject,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LocateAnythingTokenIds {
    pub box_start_token_id: u32,
    pub box_end_token_id: u32,
    pub coord_start_token_id: u32,
    pub coord_end_token_id: u32,
    pub ref_start_token_id: u32,
    pub ref_end_token_id: u32,
    pub none_token_id: u32,
    pub null_token_id: u32,
    pub im_end_token_id: u32,
    pub switch_token_id: u32,
    pub default_mask_token_id: u32,
}

impl Default for LocateAnythingTokenIds {
    fn default() -> Self {
        Self {
            box_start_token_id: 151_668,
            box_end_token_id: 151_669,
            coord_start_token_id: 151_677,
            coord_end_token_id: 152_677,
            ref_start_token_id: 151_672,
            ref_end_token_id: 151_673,
            none_token_id: 4_064,
            null_token_id: 152_678,
            im_end_token_id: 151_645,
            switch_token_id: 152_679,
            default_mask_token_id: 151_676,
        }
    }
}

impl LocateAnythingTokenIds {
    pub fn from_model_config(config: &LocateAnythingModelConfig) -> Self {
        let defaults = Self::default();
        Self {
            box_start_token_id: config.box_start_token_id,
            box_end_token_id: config.box_end_token_id,
            coord_start_token_id: config.coord_start_token_id,
            coord_end_token_id: config.coord_end_token_id,
            ref_start_token_id: config.ref_start_token_id,
            ref_end_token_id: config.ref_end_token_id,
            none_token_id: config.none_token_id,
            null_token_id: config
                .text_config
                .null_token_id
                .unwrap_or(defaults.null_token_id),
            im_end_token_id: defaults.im_end_token_id,
            switch_token_id: config
                .text_config
                .switch_token_id
                .unwrap_or(defaults.switch_token_id),
            default_mask_token_id: defaults.default_mask_token_id,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ParallelBoxDecodeConfig {
    pub keep_k: usize,
    pub start_threshold: f32,
    #[serde(default = "default_ref_start_threshold")]
    pub ref_start_threshold: f32,
    pub end_threshold: f32,
    pub generation_mode: DecodeMode,
}

fn default_ref_start_threshold() -> f32 {
    0.6
}

impl Default for ParallelBoxDecodeConfig {
    fn default() -> Self {
        Self {
            keep_k: 4,
            start_threshold: 0.7,
            ref_start_threshold: default_ref_start_threshold(),
            end_threshold: 0.2,
            generation_mode: DecodeMode::Hybrid,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ParallelBoxDecode {
    pub kind: ParallelPatternKind,
    pub tokens: Vec<u32>,
    pub need_switch_to_ar: bool,
    pub is_terminal: bool,
}

pub fn decode_parallel_box_from_logits(
    logits: &[f32],
    seq_len: usize,
    vocab_size: usize,
    token_ids: &LocateAnythingTokenIds,
    config: &ParallelBoxDecodeConfig,
) -> LocateAnythingResult<ParallelBoxDecode> {
    if logits.len() != seq_len * vocab_size {
        return Err(LocateAnythingError::Decode(format!(
            "parallel decode logits length {} does not match seq_len={seq_len} vocab_size={vocab_size}",
            logits.len()
        )));
    }
    let probs = softmax_rows(logits, seq_len, vocab_size);
    decode_parallel_box_from_probs(&probs, seq_len, vocab_size, token_ids, config)
}

pub fn decode_parallel_box_from_probs(
    probs: &[f32],
    seq_len: usize,
    vocab_size: usize,
    token_ids: &LocateAnythingTokenIds,
    config: &ParallelBoxDecodeConfig,
) -> LocateAnythingResult<ParallelBoxDecode> {
    if seq_len != 6 {
        return Err(LocateAnythingError::Decode(format!(
            "parallel box decode expects six future positions, got {seq_len}"
        )));
    }
    if probs.len() != seq_len * vocab_size {
        return Err(LocateAnythingError::Decode(format!(
            "parallel decode prob length {} does not match seq_len={seq_len} vocab_size={vocab_size}",
            probs.len()
        )));
    }
    let greedy = (0..seq_len)
        .map(|row| row_argmax(&probs[row * vocab_size..(row + 1) * vocab_size]) as u32)
        .collect::<Vec<_>>();
    if let Some(tokens) = decode_bbox_avg_from_probs(probs, vocab_size, token_ids, config) {
        return Ok(handle_parallel_pattern(
            &tokens,
            token_ids,
            config.generation_mode,
        ));
    }
    if let Some(tokens) = decode_ref_from_probs(probs, seq_len, vocab_size, token_ids, config) {
        return Ok(handle_parallel_pattern(
            &tokens,
            token_ids,
            config.generation_mode,
        ));
    }
    Ok(handle_parallel_pattern(
        &greedy,
        token_ids,
        config.generation_mode,
    ))
}

pub fn handle_parallel_pattern(
    x0: &[u32],
    token_ids: &LocateAnythingTokenIds,
    generation_mode: DecodeMode,
) -> ParallelBoxDecode {
    if x0.first() == Some(&token_ids.null_token_id)
        || x0.first() == Some(&token_ids.im_end_token_id)
    {
        return ParallelBoxDecode {
            kind: ParallelPatternKind::ImEnd,
            tokens: vec![token_ids.im_end_token_id],
            need_switch_to_ar: false,
            is_terminal: true,
        };
    }
    if x0.len() >= 2 && x0[0] == token_ids.box_start_token_id && x0[1] == token_ids.none_token_id {
        return ParallelBoxDecode {
            kind: ParallelPatternKind::EmptyBox,
            tokens: vec![
                token_ids.box_start_token_id,
                token_ids.none_token_id,
                token_ids.box_end_token_id,
            ],
            need_switch_to_ar: false,
            is_terminal: false,
        };
    }
    if x0.first() == Some(&token_ids.box_start_token_id) {
        let mut coord_ix = 1usize;
        for &coord in x0.iter().skip(1).take(4) {
            if token_ids.coord_start_token_id <= coord && coord <= token_ids.coord_end_token_id {
                coord_ix += 1;
            } else {
                break;
            }
        }
        if coord_ix == 5 && x0.get(5) == Some(&token_ids.box_end_token_id) {
            return ParallelBoxDecode {
                kind: ParallelPatternKind::CoordBox,
                tokens: x0.to_vec(),
                need_switch_to_ar: false,
                is_terminal: false,
            };
        }
        if coord_ix == 3 && x0.get(3) == Some(&token_ids.box_end_token_id) {
            return ParallelBoxDecode {
                kind: ParallelPatternKind::PointBox,
                tokens: x0[..4].to_vec(),
                need_switch_to_ar: false,
                is_terminal: false,
            };
        }
        return ParallelBoxDecode {
            kind: ParallelPatternKind::ErrorBox,
            tokens: if matches!(generation_mode, DecodeMode::ParallelBox) {
                x0.to_vec()
            } else {
                x0[..coord_ix.min(x0.len())].to_vec()
            },
            need_switch_to_ar: !matches!(generation_mode, DecodeMode::ParallelBox),
            is_terminal: false,
        };
    }

    let mut tokens = x0
        .iter()
        .copied()
        .take_while(|token| *token != token_ids.null_token_id)
        .collect::<Vec<_>>();
    if tokens.len() >= 2
        && tokens[tokens.len() - 1] == token_ids.ref_end_token_id
        && tokens[tokens.len() - 2] == token_ids.ref_end_token_id
    {
        tokens.pop();
    }
    ParallelBoxDecode {
        kind: ParallelPatternKind::RefObject,
        tokens,
        need_switch_to_ar: false,
        is_terminal: false,
    }
}

fn decode_bbox_avg_from_probs(
    probs: &[f32],
    vocab_size: usize,
    token_ids: &LocateAnythingTokenIds,
    config: &ParallelBoxDecodeConfig,
) -> Option<Vec<u32>> {
    match valid_box_frame(probs, vocab_size, token_ids, config) {
        Some(ParallelPatternKind::EmptyBox) => {
            return Some(vec![
                token_ids.box_start_token_id,
                token_ids.none_token_id,
                token_ids.box_end_token_id,
                token_ids.null_token_id,
                token_ids.null_token_id,
                token_ids.null_token_id,
            ]);
        }
        Some(ParallelPatternKind::CoordBox) => {}
        _ => return None,
    }

    let mut final_coords = Vec::with_capacity(4);
    for pos in 1..5 {
        let row = &probs[pos * vocab_size..(pos + 1) * vocab_size];
        let top = top_k_indices(row, config.keep_k);
        let valid = top
            .iter()
            .filter(|candidate| {
                token_ids.coord_start_token_id <= candidate.index as u32
                    && candidate.index as u32 <= token_ids.coord_end_token_id
            })
            .copied()
            .collect::<Vec<_>>();
        let first = valid.first()?;
        if matches!(config.generation_mode, DecodeMode::Hybrid) {
            let valid_min = valid
                .iter()
                .map(|candidate| candidate.index)
                .min()
                .unwrap_or(0);
            let valid_max = valid
                .iter()
                .map(|candidate| candidate.index)
                .max()
                .unwrap_or(0);
            let abnormal =
                first.prob < 0.9 && valid.len() > 1 && valid_max.saturating_sub(valid_min) > 60;
            final_coords.push(if abnormal { 0 } else { first.index as u32 });
        } else {
            final_coords.push(first.index as u32);
        }
    }
    let mut tokens = Vec::with_capacity(6);
    tokens.push(token_ids.box_start_token_id);
    tokens.extend(final_coords);
    tokens.push(token_ids.box_end_token_id);
    Some(tokens)
}

fn decode_ref_from_probs(
    probs: &[f32],
    seq_len: usize,
    vocab_size: usize,
    token_ids: &LocateAnythingTokenIds,
    config: &ParallelBoxDecodeConfig,
) -> Option<Vec<u32>> {
    if prob_at(probs, vocab_size, 0, token_ids.ref_start_token_id as usize)
        < config.ref_start_threshold
    {
        return None;
    }
    let mut tokens = Vec::with_capacity(seq_len);
    tokens.push(token_ids.ref_start_token_id);
    for pos in 1..seq_len {
        let row = &probs[pos * vocab_size..(pos + 1) * vocab_size];
        let top = top_k_indices(row, config.keep_k);
        let first = top.iter().find(|candidate| {
            let id = candidate.index as u32;
            id < token_ids.coord_start_token_id || id > token_ids.coord_end_token_id
        })?;
        tokens.push(first.index as u32);
    }
    Some(tokens)
}

fn valid_box_frame(
    probs: &[f32],
    vocab_size: usize,
    token_ids: &LocateAnythingTokenIds,
    config: &ParallelBoxDecodeConfig,
) -> Option<ParallelPatternKind> {
    let p_start = prob_at(probs, vocab_size, 0, token_ids.box_start_token_id as usize);
    if p_start >= config.start_threshold
        && prob_at(probs, vocab_size, 1, token_ids.none_token_id as usize) > 0.2
        && prob_at(probs, vocab_size, 2, token_ids.box_end_token_id as usize) > 0.2
        && prob_at(probs, vocab_size, 3, token_ids.null_token_id as usize) > 0.1
        && prob_at(probs, vocab_size, 4, token_ids.null_token_id as usize) > 0.1
    {
        return Some(ParallelPatternKind::EmptyBox);
    }
    let end_score = [
        token_ids.box_end_token_id,
        token_ids.null_token_id,
        token_ids.im_end_token_id,
    ]
    .into_iter()
    .map(|id| prob_at(probs, vocab_size, 5, id as usize))
    .sum::<f32>();
    (end_score >= config.end_threshold).then_some(ParallelPatternKind::CoordBox)
}

#[derive(Clone, Copy)]
struct Candidate {
    index: usize,
    prob: f32,
}

fn softmax_rows(logits: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut probs = vec![0.0; logits.len()];
    for row in 0..rows {
        let input = &logits[row * cols..(row + 1) * cols];
        let output = &mut probs[row * cols..(row + 1) * cols];
        let max = input
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (out, value) in output.iter_mut().zip(input.iter().copied()) {
            let prob = if value.is_finite() {
                (value - max).exp()
            } else {
                0.0
            };
            *out = prob;
            sum += prob;
        }
        if sum > 0.0 && sum.is_finite() {
            for out in output {
                *out /= sum;
            }
        } else if !output.is_empty() {
            let argmax = row_argmax(input);
            output[argmax] = 1.0;
        }
    }
    probs
}

fn row_argmax(row: &[f32]) -> usize {
    row.iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn top_k_indices(row: &[f32], keep_k: usize) -> Vec<Candidate> {
    if keep_k == 0 {
        return Vec::new();
    }
    let mut top = Vec::with_capacity(keep_k.min(row.len()));
    for (index, prob) in row.iter().copied().enumerate() {
        let candidate = Candidate { index, prob };
        if let Some(insert_at) = top
            .iter()
            .position(|existing| top_k_candidate_precedes(candidate, *existing))
        {
            top.insert(insert_at, candidate);
            top.truncate(keep_k);
        } else if top.len() < keep_k {
            top.push(candidate);
        }
    }
    top
}

fn top_k_candidate_precedes(candidate: Candidate, existing: Candidate) -> bool {
    match candidate.prob.total_cmp(&existing.prob) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => candidate.index < existing.index,
        std::cmp::Ordering::Less => false,
    }
}

fn prob_at(probs: &[f32], vocab_size: usize, row: usize, index: usize) -> f32 {
    probs
        .get(row * vocab_size + index)
        .copied()
        .unwrap_or_default()
}

pub fn decode_detections_from_text(
    source_query: impl Into<String>,
    text: &str,
) -> LocateAnythingResult<Vec<Detection>> {
    let source_query = source_query.into();
    let mut detections = Vec::new();
    let mut offset = 0usize;
    while let Some(start) = text[offset..].find("<box>") {
        let box_start = offset + start;
        let content_start = box_start + "<box>".len();
        let Some(end) = text[content_start..].find("</box>") else {
            return Err(LocateAnythingError::Decode(
                "found <box> without matching </box>".to_string(),
            ));
        };
        let content_end = content_start + end;
        let raw_box = &text[content_start..content_end];
        let raw_box_trimmed = raw_box.trim();
        if raw_box_trimmed.eq_ignore_ascii_case("none") {
            offset = content_end + "</box>".len();
            continue;
        }
        let values = parse_floats(raw_box);
        if values.len() != 2 && values.len() != 4 {
            return Err(LocateAnythingError::Decode(format!(
                "expected 2 point coordinates or 4 box coordinates, got {} in `{raw_box}`",
                values.len()
            )));
        }
        let label = label_before_box(&text[..box_start]).unwrap_or_else(|| source_query.clone());
        let (bbox, point) = if values.len() == 4 {
            (
                normalize_bbox([values[0], values[1], values[2], values[3]]),
                None,
            )
        } else {
            let point = normalize_point([values[0], values[1]]);
            ([point[0], point[1], point[0], point[1]], Some(point))
        };
        let mut detection = Detection {
            label,
            bbox,
            point,
            confidence: confidence_after(&text[content_end..]),
            source_query: source_query.clone(),
        };
        if let Some(point) = point_after(&text[content_end..]) {
            detection.point = Some(point);
        }
        detections.push(detection);
        offset = content_end + "</box>".len();
    }
    Ok(detections)
}

fn label_before_box(prefix: &str) -> Option<String> {
    if let Some(ref_end) = prefix.rfind("</ref>")
        && let Some(ref_start) = prefix[..ref_end].rfind("<ref>")
    {
        let candidate = prefix[ref_start + "<ref>".len()..ref_end].trim();
        if !candidate.is_empty() {
            return Some(candidate.to_string());
        }
    }
    let candidate = prefix
        .rsplit(['\n', ';'])
        .next()
        .unwrap_or(prefix)
        .trim()
        .trim_matches([':', '-', ' ', '\t']);
    if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_string())
    }
}

fn point_after(suffix: &str) -> Option<[f32; 2]> {
    let start = suffix.find("<point>")?;
    let content_start = start + "<point>".len();
    let end = suffix[content_start..].find("</point>")?;
    let values = parse_floats(&suffix[content_start..content_start + end]);
    if values.len() == 2 {
        Some(normalize_point([values[0], values[1]]))
    } else {
        None
    }
}

fn confidence_after(suffix: &str) -> Option<f32> {
    let index = suffix.find("confidence")?;
    let values = parse_floats(&suffix[index..suffix.len().min(index + 48)]);
    values.into_iter().find(|value| value.is_finite())
}

fn parse_floats(value: &str) -> Vec<f32> {
    value
        .split(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+' | 'e' | 'E')))
        .filter(|part| !part.trim().is_empty())
        .filter_map(|part| part.parse::<f32>().ok())
        .map(|value| if value > 1.0 { value / 1000.0 } else { value })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_box_point_and_confidence() {
        let detections = decode_detections_from_text(
            "find chairs",
            "chair: <box>100, 200, 300, 850</box> <point>200, 820</point> confidence=0.91",
        )
        .unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, "chair");
        assert_eq!(detections[0].bbox, [0.1, 0.2, 0.3, 0.85]);
        assert_eq!(detections[0].point, Some([0.2, 0.82]));
        assert!((detections[0].confidence.unwrap() - 0.91).abs() < 1.0e-6);
    }

    #[test]
    fn decodes_upstream_ref_box_format() {
        let detections = decode_detections_from_text(
            "find chairs",
            "<ref>chair</ref><box><100><200><300><850></box><ref>table</ref><box><0><10><1000><900></box>",
        )
        .unwrap();
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].label, "chair");
        assert_eq!(detections[0].bbox, [0.1, 0.2, 0.3, 0.85]);
        assert_eq!(detections[1].label, "table");
        assert_eq!(detections[1].bbox, [0.0, 0.01, 1.0, 0.9]);
    }

    #[test]
    fn decodes_upstream_point_and_none_format() {
        let detections = decode_detections_from_text(
            "point to search",
            "<ref>search button</ref><box><250><750></box><ref>missing</ref><box>none</box>",
        )
        .unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, "search button");
        assert_eq!(detections[0].bbox, [0.25, 0.75, 0.25, 0.75]);
        assert_eq!(detections[0].point, Some([0.25, 0.75]));
    }

    #[test]
    fn rejects_malformed_boxes() {
        let err = decode_detections_from_text("find table", "table <box>0.1,0.2,0.3</box>")
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 2 point coordinates or 4 box coordinates"));
    }

    #[test]
    fn decodes_parallel_coord_box_from_probs() {
        let ids = LocateAnythingTokenIds {
            box_start_token_id: 1,
            box_end_token_id: 2,
            coord_start_token_id: 10,
            coord_end_token_id: 20,
            ref_start_token_id: 3,
            ref_end_token_id: 4,
            none_token_id: 5,
            null_token_id: 6,
            im_end_token_id: 7,
            switch_token_id: 8,
            default_mask_token_id: 9,
        };
        let vocab = 32;
        let mut probs = vec![0.0; 6 * vocab];
        for (row, token) in [1, 11, 12, 13, 14, 2].into_iter().enumerate() {
            probs[row * vocab + token] = 1.0;
        }
        let decoded =
            decode_parallel_box_from_probs(&probs, 6, vocab, &ids, &Default::default()).unwrap();
        assert_eq!(decoded.kind, ParallelPatternKind::CoordBox);
        assert_eq!(decoded.tokens, vec![1, 11, 12, 13, 14, 2]);
        assert!(!decoded.need_switch_to_ar);
    }

    #[test]
    fn decodes_parallel_empty_box_from_probs() {
        let ids = LocateAnythingTokenIds {
            box_start_token_id: 1,
            box_end_token_id: 2,
            coord_start_token_id: 10,
            coord_end_token_id: 20,
            ref_start_token_id: 3,
            ref_end_token_id: 4,
            none_token_id: 5,
            null_token_id: 6,
            im_end_token_id: 7,
            switch_token_id: 8,
            default_mask_token_id: 9,
        };
        let vocab = 32;
        let mut probs = vec![0.0; 6 * vocab];
        probs[ids.box_start_token_id as usize] = 0.95;
        probs[vocab + ids.none_token_id as usize] = 0.9;
        probs[2 * vocab + ids.box_end_token_id as usize] = 0.9;
        probs[3 * vocab + ids.null_token_id as usize] = 0.9;
        probs[4 * vocab + ids.null_token_id as usize] = 0.9;
        probs[5 * vocab + ids.box_end_token_id as usize] = 0.9;
        let decoded =
            decode_parallel_box_from_probs(&probs, 6, vocab, &ids, &Default::default()).unwrap();
        assert_eq!(decoded.kind, ParallelPatternKind::EmptyBox);
        assert_eq!(decoded.tokens, vec![1, 5, 2]);
    }

    #[test]
    fn top_k_indices_preserves_descending_prob_and_stable_index_ties() {
        let row = [0.1, 0.9, 0.4, 0.9, 0.2, 0.8];
        let top = top_k_indices(&row, 4)
            .into_iter()
            .map(|candidate| candidate.index)
            .collect::<Vec<_>>();
        assert_eq!(top, vec![1, 3, 5, 2]);
    }
}
