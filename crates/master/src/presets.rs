use anyhow::{Context, Result, bail};

use crate::config::ConfigFile;

pub const PRESET_NAMES: [&str; 4] = ["neutral", "gentle", "loud", "vintage"];

pub fn preset(name: &str) -> Result<ConfigFile> {
    let source: &str = match name {
        "neutral" => include_str!("../presets/neutral.toml"),
        "gentle" => include_str!("../presets/gentle.toml"),
        "loud" => include_str!("../presets/loud.toml"),
        "vintage" => include_str!("../presets/vintage.toml"),
        other => bail!(
            "Unknown preset '{other}'. Available presets: {}",
            PRESET_NAMES.join(", ")
        ),
    };

    toml::from_str(source).with_context(|| format!("Built-in preset '{name}' is invalid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_preset_parses() {
        for name in PRESET_NAMES {
            preset(name).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        }
    }
}
