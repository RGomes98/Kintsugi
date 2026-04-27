use std::path::Path;

use anyhow::{Context, Result, bail};
use tch::{CModule, Device, Kind, Tensor, no_grad};

use crate::{DeviceChoice, config::TARGET_FRAMES};

pub(crate) fn load_model(
    models_dir: &Path,
    file_name: &'static str,
    device: Device,
) -> Result<CModule> {
    let model_path: std::path::PathBuf = models_dir.join(file_name);
    CModule::load_on_device(&model_path, device)
        .with_context(|| format!("Failed to load model from {}", model_path.display()))
}

pub(crate) fn forward_chunk(model: &CModule, input: &Tensor, device: Device) -> Result<Tensor> {
    let input_device: Tensor = input.to_device(device).contiguous();
    let output: Tensor = no_grad(|| model.forward_ts(&[input_device]))
        .context("Model forward pass failed")?
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous();
    Ok(output)
}

pub(crate) fn validate_output_shape(output: &Tensor) -> Result<()> {
    let shape: Vec<i64> = output.size();
    if shape.as_slice() != [1, 4, 2, TARGET_FRAMES as i64] {
        bail!(
            "Unexpected model output shape {:?}, expected [1, 4, 2, {}]",
            shape,
            TARGET_FRAMES
        );
    }
    Ok(())
}

pub(crate) fn resolve_device_context(
    device_choice: DeviceChoice,
) -> Result<(Device, &'static str)> {
    match device_choice {
        DeviceChoice::Cpu => Ok((Device::Cpu, crate::config::MODEL_FILENAME_CPU)),
        DeviceChoice::Cuda => {
            if tch::utils::has_cuda() {
                Ok((
                    Device::cuda_if_available(),
                    crate::config::MODEL_FILENAME_CUDA,
                ))
            } else {
                bail!(
                    "CUDA was requested, but no compatible NVIDIA GPU or CUDA drivers were detected."
                )
            }
        }
    }
}
