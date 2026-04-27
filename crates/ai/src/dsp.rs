pub fn build_demucs_weight_window(size: usize, transition_power: f32) -> Vec<f32> {
    assert!(size > 0);
    assert!(transition_power >= 1.0);

    let left_len: usize = size / 2;
    let right_len: usize = size - left_len;

    let mut weight: Vec<f32> = Vec::with_capacity(size);

    for i in 0..left_len {
        weight.push((i + 1) as f32);
    }

    for i in 0..right_len {
        weight.push((right_len - i) as f32);
    }

    let max_w: f32 = weight.iter().copied().fold(0.0f32, f32::max).max(1.0);

    weight
        .into_iter()
        .map(|w: f32| (w / max_w).powf(transition_power))
        .collect()
}
