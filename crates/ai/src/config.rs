pub(crate) const MODEL_FILENAME_CUDA: &str = "htdemucs_cuda.pt";
pub(crate) const MODEL_FILENAME_CPU: &str = "htdemucs.pt";
pub(crate) const MODEL_SAMPLE_RATE: u32 = 44_100;
pub(crate) const TARGET_FRAMES: usize = 343_980;

#[derive(Clone, Debug)]
pub enum DeviceChoice {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy)]
pub struct ExtractOptions {
    pub shifts: usize,
    pub overlap: f32,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            shifts: 1,
            overlap: 0.5,
        }
    }
}

impl ExtractOptions {
    pub fn validate(self) -> anyhow::Result<Self> {
        if self.shifts == 0 {
            anyhow::bail!("Shifts must be >= 1");
        }

        if !(0.0..1.0).contains(&self.overlap) {
            anyhow::bail!("Overlap must be in the range [0.0, 1.0)");
        }

        Ok(self)
    }
}
