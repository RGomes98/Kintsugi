pub const N_CHANNELS: usize = 2;
pub const N_STEMS: usize = 4;

#[derive(Clone, Debug)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Clone, Debug)]
pub struct Stems {
    pub drums: AudioBuffer,
    pub bass: AudioBuffer,
    pub other: AudioBuffer,
    pub vocals: AudioBuffer,
}
