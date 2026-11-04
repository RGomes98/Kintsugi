use std::path::{Path, PathBuf};

use ai::{DeviceChoice, ExtractOptions};
use anyhow::{Context, Result};
use clap::builder::PossibleValuesParser;
use clap::{Args, Parser, Subcommand, ValueEnum};
use dsp::loudness::{LoudnessStats, MIN_MEASURABLE_SECONDS, format_db, measure_loudness};

const MODELS_DIR: &str = "python/models";
const DEFAULT_OUTPUT_DIR: &str = "output";

#[derive(Parser)]
#[command(version, about = "Audio separation and remastering")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Split a track into drums, bass, vocals and other instruments
    Separate(SeparateArgs),
    /// Remaster a track: EQ, compression, loudness normalization and limiting
    Master(MasterArgs),
    /// Measure loudness, loudness range and peaks without changing the audio
    Analyze(AnalyzeArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum Device {
    Cpu,
    Cuda,
}

#[derive(Args)]
struct SeparateArgs {
    /// Input audio file
    input: PathBuf,

    #[arg(long, value_enum, default_value_t = Device::Cpu)]
    device: Device,

    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    output_dir: PathBuf,
}

#[derive(Args)]
struct MasterArgs {
    /// Input audio file (stereo)
    input: PathBuf,

    /// Settings to start from [default: neutral]
    #[arg(long, value_parser = PossibleValuesParser::new(master::PRESET_NAMES))]
    preset: Option<String>,

    /// TOML file whose values override the preset
    #[arg(long)]
    config: Option<PathBuf>,

    /// Loudness target in LUFS; overrides the preset and the config file
    #[arg(long, allow_negative_numbers = true)]
    target_lufs: Option<f32>,

    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    output_dir: PathBuf,
}

#[derive(Args)]
struct AnalyzeArgs {
    /// Input audio file (stereo)
    input: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Separate(args) => run_separate(args),
        Command::Master(args) => run_master(args),
        Command::Analyze(args) => run_analyze(args),
    }
}

fn run_separate(args: SeparateArgs) -> Result<()> {
    let device_choice: DeviceChoice = match args.device {
        Device::Cpu => DeviceChoice::Cpu,
        Device::Cuda => DeviceChoice::Cuda,
    };

    let models_dir: &Path = Path::new(MODELS_DIR);

    println!("Decoding audio file...");
    let buffer: kintsugi_core::AudioBuffer = io::decode_audio(&args.input)?;

    println!("Extracting stems...");
    let stems: kintsugi_core::Stems = ai::extract_stems(
        &buffer,
        models_dir,
        &ExtractOptions::default(),
        device_choice,
    )?;

    println!("\nSaving output files...");
    io::write_wav(&args.output_dir.join("drums.wav"), &stems.drums)?;
    io::write_wav(&args.output_dir.join("bass.wav"), &stems.bass)?;
    io::write_wav(&args.output_dir.join("vocals.wav"), &stems.vocals)?;
    io::write_wav(&args.output_dir.join("other_instruments.wav"), &stems.other)?;

    println!("All done!");
    Ok(())
}

fn run_master(args: MasterArgs) -> Result<()> {
    let mut config: master::MasterConfig =
        master::load_config(args.preset.as_deref(), args.config.as_deref())?;

    if let Some(target_lufs) = args.target_lufs {
        config.loudness.enabled = true;
        config.loudness.target_lufs = target_lufs;
    }

    println!("Decoding audio file...");
    let buffer: kintsugi_core::AudioBuffer = io::decode_audio(&args.input)?;

    println!("Mastering...");
    let outcome: master::MasterOutcome = master::master_audio(&buffer, &config)?;

    let output_path: PathBuf = args
        .output_dir
        .join(format!("{}_mastered.wav", file_stem(&args.input)?));
    io::write_wav(&output_path, &outcome.audio)?;

    println!("\n{}", outcome.report);
    println!("\nSaved {}", output_path.display());
    Ok(())
}

fn run_analyze(args: AnalyzeArgs) -> Result<()> {
    let buffer: kintsugi_core::AudioBuffer = io::decode_audio(&args.input)?;
    let stats: LoudnessStats = measure_loudness(&buffer)?;

    if stats.is_silent() {
        println!(
            "Integrated loudness : not measurable (silent or shorter than {MIN_MEASURABLE_SECONDS} s)"
        );
    } else {
        println!(
            "Integrated loudness : {} LUFS",
            format_db(stats.integrated_lufs)
        );
        println!("Loudness range      : {:.1} LU", stats.loudness_range_lu);
    }
    println!(
        "True peak           : {} dBTP",
        format_db(stats.true_peak_dbtp)
    );
    println!(
        "Sample peak         : {} dBFS",
        format_db(stats.sample_peak_db)
    );
    Ok(())
}

fn file_stem(path: &Path) -> Result<&str> {
    path.file_stem()
        .context("Input path does not have a valid file name")?
        .to_str()
        .context("Input path contains invalid UTF-8 characters")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("cli").chain(args.iter().copied()))
    }

    #[test]
    fn separate_defaults_to_cpu_and_the_output_directory() {
        let cli: Cli = parse(&["separate", "song.mp3"]).ok().unwrap();
        let Command::Separate(args) = cli.command else {
            panic!("expected separate");
        };
        assert!(matches!(args.device, Device::Cpu));
        assert_eq!(args.output_dir, PathBuf::from("output"));
    }

    #[test]
    fn separate_accepts_cuda() {
        let cli: Cli = parse(&["separate", "song.mp3", "--device", "cuda"])
            .ok()
            .unwrap();
        let Command::Separate(args) = cli.command else {
            panic!("expected separate");
        };
        assert!(matches!(args.device, Device::Cuda));
    }

    #[test]
    fn master_accepts_a_negative_loudness_target() {
        let cli: Cli = parse(&["master", "song.mp3", "--target-lufs", "-12.5"])
            .ok()
            .unwrap();
        let Command::Master(args) = cli.command else {
            panic!("expected master");
        };
        assert_eq!(args.target_lufs, Some(-12.5));
    }

    #[test]
    fn master_rejects_an_unknown_preset() {
        assert!(parse(&["master", "song.mp3", "--preset", "warm"]).is_err());
    }

    #[test]
    fn master_accepts_every_builtin_preset() {
        for name in master::PRESET_NAMES {
            assert!(parse(&["master", "song.mp3", "--preset", name]).is_ok());
        }
    }

    #[test]
    fn the_old_positional_form_is_rejected() {
        assert!(parse(&["song.mp3", "cpu"]).is_err());
    }
}
