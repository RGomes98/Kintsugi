use ai::{DeviceChoice, ExtractOptions};
use anyhow::{Result, bail};
use std::{env, path::Path};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        bail!("Usage: cargo run --release --bin cli <input_file> [cpu|cuda]");
    }

    let input_path: &Path = Path::new(&args[1]);

    let device_choice: DeviceChoice = match args[2].to_lowercase().as_str() {
        "cpu" => DeviceChoice::Cpu,
        "cuda" => DeviceChoice::Cuda,
        _ => bail!("Invalid device. Use 'cpu' or 'cuda'"),
    };

    let models_dir: &Path = Path::new("python/models");

    println!("Decoding audio file...");
    let buffer: core::AudioBuffer = io::decode_audio(input_path)?;

    println!("Extracting stems...");
    let stems: core::Stems = ai::extract_stems(
        &buffer,
        models_dir,
        &ExtractOptions::default(),
        device_choice,
    )?;

    println!("\nSaving output files...");
    io::write_wav(Path::new("output/drums.wav"), &stems.drums)?;
    io::write_wav(Path::new("output/bass.wav"), &stems.bass)?;
    io::write_wav(Path::new("output/vocals.wav"), &stems.vocals)?;
    io::write_wav(Path::new("output/other_instruments.wav"), &stems.other)?;

    println!("All done!");
    Ok(())
}
