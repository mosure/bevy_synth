use burn::prelude::*;

use burn_tripo::model::triposg::image_encoder::DinoImageProcessor;

#[test]
fn dino_preprocess_resizes_to_model_size() {
    let processor = DinoImageProcessor {
        do_resize: true,
        size_shortest_edge: Some(256),
        do_center_crop: true,
        crop_size: Some([224, 224]),
        do_rescale: false,
        do_normalize: false,
        ..Default::default()
    };

    let device = Default::default();
    let input = Tensor::<burn::backend::NdArray<f32>, 4>::zeros([1, 3, 2044, 2044], &device);
    let output = processor.preprocess(input);
    let [_batch, _channels, height, width] = output.shape().dims();
    assert_eq!(height, 224);
    assert_eq!(width, 224);
}
