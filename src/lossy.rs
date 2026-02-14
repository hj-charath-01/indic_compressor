// src/lossy.rs - Lossy compression for Indic text
// Phonetically-aware lossy compression

use serde::{Deserialize, Serialize};

/// Quality levels for lossy compression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityLevel {
    Lossless = 0,
    High = 1,
    Medium = 2,
    Low = 3,
}

impl QualityLevel {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => QualityLevel::Lossless,
            1 => QualityLevel::High,
            2 => QualityLevel::Medium,
            _ => QualityLevel::Low,
        }
    }
}

/// Metadata for lossy compression
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LossyMetadata {
    pub quality: QualityLevel,
    pub original_size: usize,
    pub compressed_size: usize,
    pub script: String,
}

impl LossyMetadata {
    pub fn new(quality: QualityLevel, original: &str, compressed: &str, script: &str) -> Self {
        Self {
            quality,
            original_size: original.len(),
            compressed_size: compressed.len(),
            script: script.to_string(),
        }
    }
    
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Self {
        serde_json::from_slice(bytes).unwrap_or(Self {
            quality: QualityLevel::Lossless,
            original_size: 0,
            compressed_size: 0,
            script: String::new(),
        })
    }
}

/// Lossy compressor for Indic text
pub struct LossyCompressor {
    quality: QualityLevel,
}

impl LossyCompressor {
    pub fn new(quality: QualityLevel) -> Self {
        Self { quality }
    }
    
    /// Compress text with lossy transformations
    pub fn compress(&self, text: &str) -> String {
        match self.quality {
            QualityLevel::Lossless => text.to_string(),
            QualityLevel::High => self.compress_high(text),
            QualityLevel::Medium => self.compress_medium(text),
            QualityLevel::Low => self.compress_low(text),
        }
    }
    
    /// High quality: remove only redundant whitespace
    fn compress_high(&self, text: &str) -> String {
        // Normalize whitespace
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Medium quality: normalize whitespace and some punctuation
    fn compress_medium(&self, text: &str) -> String {
        let normalized = self.compress_high(text);
        
        // Normalize multiple punctuation
        normalized
            .replace("...", "…")
            .replace("!!", "!")
            .replace("??", "?")
    }
    
    /// Low quality: aggressive normalization
    fn compress_low(&self, text: &str) -> String {
        let normalized = self.compress_medium(text);
        
        // Remove some punctuation
        normalized
            .replace(',', "")
            .replace(';', "")
    }
    
    /// Estimate savings from lossy compression
    pub fn estimate_savings(&self, text: &str) -> f64 {
        let original_len = text.len() as f64;
        if original_len == 0.0 {
            return 0.0;
        }
        
        let compressed = self.compress(text);
        let compressed_len = compressed.len() as f64;
        
        ((original_len - compressed_len) / original_len) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lossless() {
        let compressor = LossyCompressor::new(QualityLevel::Lossless);
        let text = "Hello   world  !";
        let compressed = compressor.compress(text);
        
        assert_eq!(text, compressed);
    }
    
    #[test]
    fn test_high_quality() {
        let compressor = LossyCompressor::new(QualityLevel::High);
        let text = "Hello   world  !";
        let compressed = compressor.compress(text);
        
        assert_eq!(compressed, "Hello world !");
    }
    
    #[test]
    fn test_estimate_savings() {
        let compressor = LossyCompressor::new(QualityLevel::High);
        let text = "Hello   world  !  How   are   you?";
        let savings = compressor.estimate_savings(text);
        
        assert!(savings > 0.0);
        assert!(savings < 100.0);
    }
    
    #[test]
    fn test_metadata() {
        let meta = LossyMetadata::new(
            QualityLevel::High,
            "original text",
            "compressed",
            "devanagari"
        );
        
        let bytes = meta.to_bytes();
        let restored = LossyMetadata::from_bytes(&bytes);
        
        assert_eq!(restored.quality, QualityLevel::High);
        assert_eq!(restored.script, "devanagari");
    }
}