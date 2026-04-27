use std::env;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use core::AudioBuffer;

pub fn run_pipeline() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        bail!("Usage: cargo run --bin cli <input_file>");
    }

    let input_path: &Path = Path::new(&args[1]);

    let input_prefix: &str = input_path
        .file_stem()
        .context("Input path does not have a valid file name")?
        .to_str()
        .context("Input path contains invalid UTF-8 characters")?;

    let file_name: &String = &format!("{input_prefix}.wav");
    let output_path: &Path = Path::new(file_name);

    let buffer: AudioBuffer = decode_audio(input_path)?;
    write_wav(output_path, &buffer)?;
    Ok(())
}

pub fn decode_audio(path: &Path) -> Result<AudioBuffer> {
    let mut hint: Hint = Hint::new();

    if let Some(ext) = path.extension().and_then(|e: &std::ffi::OsStr| e.to_str()) {
        hint.with_extension(ext);
    }

    let source: Box<File> = Box::new(File::open(path).context("Failed to open input file")?);
    let mss: MediaSourceStream = MediaSourceStream::new(source, Default::default());

    let mut probe: symphonia::core::probe::ProbeResult = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Failed to probe audio format")?;

    let track: &symphonia::core::formats::Track = probe
        .format
        .default_track()
        .context("No default audio track found")?;

    let sample_rate: u32 = track
        .codec_params
        .sample_rate
        .context("Unknown sample rate")?;

    let channels: u16 = track
        .codec_params
        .channels
        .context("Unknown channel count")?
        .count() as u16;

    let mut decoder: Box<dyn Decoder> = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to initialize decoder")?;

    let mut samples: Vec<f32> = Vec::new();
    let track_id: u32 = track.id;

    loop {
        let packet: symphonia::core::formats::Packet = match probe.format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break,
            Err(Error::DecodeError(_)) => continue,
            Err(e) => bail!("Unexpected error reading packet: {e}"),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let mut sample_buf: SampleBuffer<f32> =
                    SampleBuffer::<f32>::new(audio_buf.capacity() as u64, *audio_buf.spec());

                sample_buf.copy_interleaved_ref(audio_buf);
                samples.extend_from_slice(sample_buf.samples());
            }
            Err(Error::DecodeError(_)) => continue,
            Err(e) => bail!("Unexpected decoder error: {e}"),
        }
    }

    Ok(AudioBuffer {
        samples,
        channels,
        sample_rate,
    })
}

pub fn write_wav(path: &Path, buffer: &AudioBuffer) -> Result<()> {
    ensure_dir_exists(path)?;

    let spec: hound::WavSpec = hound::WavSpec {
        channels: buffer.channels,
        sample_rate: buffer.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer: hound::WavWriter<std::io::BufWriter<File>> =
        hound::WavWriter::create(path, spec).context("Failed to create output file")?;

    for &sample in &buffer.samples {
        writer
            .write_sample(sample)
            .context("Failed to write sample")?;
    }

    writer.finalize().context("Failed to finalize WAV file")?;
    Ok(())
}

pub fn ensure_dir_exists(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create directory structure for {}",
                parent.display()
            )
        })?;
    }
    Ok(())
}
