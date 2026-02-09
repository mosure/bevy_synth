use std::path::PathBuf;

use burn_trellis::preprocess::{PreprocessConfig, preprocess_image_path};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_trellis_preprocess(c: &mut Criterion) {
    let Some(input_image) = std::env::var("TRELLIS2_BENCH_IMAGE").ok() else {
        eprintln!("Skipping bench: TRELLIS2_BENCH_IMAGE is unset.");
        return;
    };

    let input = PathBuf::from(input_image);
    if !input.exists() {
        eprintln!("Skipping bench: TRELLIS2_BENCH_IMAGE does not exist.");
        return;
    }

    let mut group = c.benchmark_group("trellis2_preprocess");
    group.bench_function("default", |b| {
        b.iter(|| {
            let _ = preprocess_image_path(input.as_path(), PreprocessConfig::default());
        })
    });
    group.finish();
}

criterion_group!(benches, bench_trellis_preprocess);
criterion_main!(benches);
