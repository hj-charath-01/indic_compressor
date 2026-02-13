// src/lib.rs - ENHANCED with ACTUAL Neural-Hybrid Integration
// Patent-worthy features:
// 1. Neural-enhanced entropy coding with script-specific embeddings
// 2. Phonetically-aware lossy compression for Indic scripts

pub mod dict;
pub mod tokenize;
pub mod chunk;
pub mod ppm;
pub mod neural;
pub mod lossy;
pub mod chunk_neural;

use anyhow::{Result, anyhow};
use wasm_bindgen::prelude::*;
use std::convert::TryInto;
use lossy::{LossyCompressor, QualityLevel, LossyMetadata};
use neural::load_pretrained_model;

/// Minimum text size (in bytes) before attempting compression
const COMPRESSION_THRESHOLD: usize = 500;

/// Compression options for advanced features
#[derive(Debug, Clone)]
pub struct CompressionOptions {
    /// Chunk size (tokens per chunk)
    pub chunk_size: usize,
    /// Enable neural-hybrid compression
    pub use_neural: bool,
    /// Neural weight (0.0 = pure PPM, 1.0 = pure neural)
    pub neural_weight: f32,
    /// Lossy compression quality level
    pub quality: QualityLevel,
    /// Target script for optimization
    pub script: String,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            chunk_size: 40,
            use_neural: false,
            neural_weight: 0.3,
            quality: QualityLevel::Lossless,
            script: "devanagari".to_string(),
        }
    }
}

/// Enhanced encoding with ACTUAL neural-hybrid and lossy compression
pub fn encode_stream_advanced(text: &str, options: CompressionOptions) -> Result<Vec<u8>> {
    let text_bytes = text.as_bytes().len();
    
    // Step 1: Apply lossy compression if requested
    let (processed_text, lossy_meta) = if options.quality != QualityLevel::Lossless {
        let compressor = LossyCompressor::new(options.quality);
        let compressed = compressor.compress(text);
        
        let original_bytes = text.as_bytes().len();
        let compressed_bytes = compressed.as_bytes().len();
        
        if compressed_bytes < original_bytes {
            let meta = LossyMetadata::new(options.quality, text, &compressed, &options.script);
            (compressed, Some(meta))
        } else {
            (text.to_string(), None)
        }
    } else {
        (text.to_string(), None)
    };
    
    // For very small texts, use uncompressed format
    if text_bytes < COMPRESSION_THRESHOLD {
        return encode_uncompressed(&processed_text, lossy_meta.as_ref());
    }
    
    // Step 2: Adaptive chunking
    let actual_chunk_size = if text_bytes < 2000 {
        200
    } else if text_bytes < 10000 {
        100
    } else {
        options.chunk_size
    };
    
    // Step 3: Load neural model if requested (ACTUALLY USED NOW!)
    let mut neural_model = if options.use_neural {
        let model = load_pretrained_model(&options.script);
        if model.is_none() {
            // Model failed to load - will fall back to standard PPM
            // You can add logging here: eprintln!("Warning: Neural model failed to load for script: {}", options.script);
        }
        model
    } else {
        None
    };
    
    // Step 4: Tokenize and compress WITH NEURAL
    let tokens = tokenize::tokenize(&processed_text);
    let mut dict = dict::MultiTierDict::new();
    let mut stream = chunk::Stream::new();

    // ACTUAL neural integration!
    for chunk_tokens in tokens.chunks(actual_chunk_size) {
        let ch = if neural_model.is_some() {
            // Use neural-hybrid encoding (model successfully loaded)
            chunk::encode_chunk_with_neural(
                &mut dict, 
                chunk_tokens.to_vec(),
                neural_model.as_mut(),
                options.neural_weight,
            )?
        } else {
            // Standard PPM encoding (model not loaded or disabled)
            chunk::encode_chunk(&mut dict, chunk_tokens.to_vec())?
        };
        stream.chunks.push(ch);
    }

    // Step 5: Write stream with metadata
    let mut stream_bytes = chunk::write_stream(&stream)?;
    
    // Prepend lossy metadata if present
    if let Some(meta) = lossy_meta {
        let meta_bytes = meta.to_bytes();
        let mut final_bytes = Vec::new();
        final_bytes.extend(b"IL"); // "IL" = Indic Lossy
        final_bytes.push(meta_bytes.len() as u8);
        final_bytes.extend(meta_bytes);
        final_bytes.extend(stream_bytes);
        stream_bytes = final_bytes;
    }
    
    Ok(stream_bytes)
}

/// Original encoding function (backward compatibility)
pub fn encode_stream(text: &str, chunk_size: usize) -> Result<Vec<u8>> {
    let options = CompressionOptions {
        chunk_size,
        ..Default::default()
    };
    encode_stream_advanced(text, options)
}

/// Store text uncompressed with optional lossy metadata
fn encode_uncompressed(text: &str, lossy_meta: Option<&LossyMetadata>) -> Result<Vec<u8>> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    
    if len > u32::MAX as usize {
        return Err(anyhow!("Text too large for uncompressed format"));
    }
    
    let mut out = Vec::with_capacity(6 + len);
    
    if let Some(meta) = lossy_meta {
        out.extend_from_slice(b"UL");
        let meta_bytes = meta.to_bytes();
        out.push(meta_bytes.len() as u8);
        out.extend(meta_bytes);
    } else {
        out.extend_from_slice(b"UC");
    }
    
    out.extend_from_slice(&(len as u32).to_be_bytes());
    out.extend_from_slice(bytes);
    
    Ok(out)
}

/// Decode with support for all formats (compressed, uncompressed, lossy)
/// Now supports neural-compressed data too!
pub fn decode_full(data: &[u8]) -> Result<String> {
    decode_full_with_options(data, None, 0.0)
}

/// Decode with neural options (for symmetry with encoding)
pub fn decode_full_with_options(
    data: &[u8],
    neural_model: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    if data.len() < 2 {
        return Err(anyhow!("Data too short to be valid"));
    }
    
    let magic = &data[0..2];
    
    match magic {
        b"UC" => decode_uncompressed(data),
        b"UL" => decode_uncompressed_lossy(data),
        b"IC" => decode_compressed_with_neural(data, neural_model, alpha),
        b"IL" => decode_compressed_lossy_with_neural(data, neural_model, alpha),
        _ => Err(anyhow!("Unknown format - magic bytes: {:?}", magic)),
    }
}

/// Decode uncompressed format with lossy metadata
fn decode_uncompressed_lossy(data: &[u8]) -> Result<String> {
    if data.len() < 7 {
        return Err(anyhow!("Invalid UL format - too short"));
    }
    
    let meta_len = data[2] as usize;
    if data.len() < 3 + meta_len + 4 {
        return Err(anyhow!("Invalid UL format - truncated"));
    }
    
    let _meta = LossyMetadata::from_bytes(&data[3..3 + meta_len]);
    let data_start = 3 + meta_len;
    
    let size = u32::from_be_bytes(data[data_start..data_start + 4].try_into()?) as usize;
    
    if data.len() < data_start + 4 + size {
        return Err(anyhow!("Invalid UL format - data truncated"));
    }
    
    let text = String::from_utf8(data[data_start + 4..data_start + 4 + size].to_vec())?;
    Ok(text)
}

/// Decode compressed format with lossy metadata and neural support
fn decode_compressed_lossy_with_neural(
    data: &[u8],
    neural_model: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    if data.len() < 7 {
        return Err(anyhow!("Invalid IL format - too short"));
    }
    
    let meta_len = data[2] as usize;
    if data.len() < 3 + meta_len {
        return Err(anyhow!("Invalid IL format - truncated"));
    }
    
    let _meta = LossyMetadata::from_bytes(&data[3..3 + meta_len]);
    let compressed_data = &data[3 + meta_len..];
    
    decode_compressed_with_neural(compressed_data, neural_model, alpha)
}

/// Decode uncompressed format
fn decode_uncompressed(data: &[u8]) -> Result<String> {
    if data.len() < 6 {
        return Err(anyhow!("Invalid uncompressed format - too short"));
    }
    
    let size = u32::from_be_bytes(data[2..6].try_into()?) as usize;
    
    if data.len() < 6 + size {
        return Err(anyhow!("Invalid uncompressed format - truncated data"));
    }
    
    let text = String::from_utf8(data[6..6+size].to_vec())?;
    Ok(text)
}

/// Decode compressed format with neural support
fn decode_compressed_with_neural(
    data: &[u8],
    mut neural_model: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    let mut dict = dict::MultiTierDict::new();
    let stream = chunk::read_stream(data)?;
    let mut out = String::new();

    for ch in stream.chunks {
        let part = chunk::decode_chunk_with_neural(&mut dict, &ch, neural_model.as_deref_mut(), alpha)?;
        out.push_str(&part);
    }
    
    Ok(out)
}

/// Decode prefix with format detection
pub fn decode_prefix(data: &[u8], upto: usize) -> Result<String> {
    decode_prefix_with_options(data, upto, None, 0.0)
}

/// Decode prefix with neural options
pub fn decode_prefix_with_options(
    data: &[u8],
    upto: usize,
    neural_model: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    if data.len() < 2 {
        return Err(anyhow!("Data too short to be valid"));
    }
    
    let magic = &data[0..2];
    
    match magic {
        b"UC" | b"UL" => {
            decode_full_with_options(data, neural_model, alpha)
        }
        b"IC" => {
            decode_compressed_prefix_with_neural(data, upto, neural_model, alpha)
        }
        b"IL" => {
            if data.len() < 7 {
                return Err(anyhow!("Invalid IL format"));
            }
            let meta_len = data[2] as usize;
            let compressed_data = &data[3 + meta_len..];
            decode_compressed_prefix_with_neural(compressed_data, upto, neural_model, alpha)
        }
        _ => Err(anyhow!("Unknown format")),
    }
}

fn decode_compressed_prefix_with_neural(
    data: &[u8],
    upto: usize,
    mut neural_model: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    let mut dict = dict::MultiTierDict::new();
    let stream = chunk::read_stream(data)?;
    let mut out = String::new();

    for (i, ch) in stream.chunks.iter().enumerate() {
        if i >= upto { break; }
        let part = chunk::decode_chunk_with_neural(&mut dict, &ch, neural_model.as_deref_mut(), alpha)?;
        out.push_str(&part);
    }
    
    Ok(out)
}

/// Get compression statistics with neural/lossy info
pub fn get_compression_stats(text: &str, options: CompressionOptions) -> Result<CompressionStats> {
    let original_size = text.as_bytes().len();
    let encoded = encode_stream_advanced(text, options.clone())?;
    let compressed_size = encoded.len();
    
    let ratio = if original_size > 0 {
        (compressed_size as f64 / original_size as f64) * 100.0
    } else {
        0.0
    };
    
    let format = if encoded.len() >= 2 {
        match &encoded[0..2] {
            b"UC" => "uncompressed",
            b"UL" => "uncompressed-lossy",
            b"IC" => "compressed",
            b"IL" => "compressed-lossy",
            _ => "unknown",
        }
    } else {
        "unknown"
    };
    
    let lossy_savings = if options.quality != QualityLevel::Lossless {
        let compressor = LossyCompressor::new(options.quality);
        compressor.estimate_savings(text)
    } else {
        0.0
    };
    
    Ok(CompressionStats {
        original_size,
        compressed_size,
        ratio,
        format: format.to_string(),
        neural_enabled: options.use_neural,
        lossy_savings,
        quality: options.quality,
    })
}

#[derive(Debug)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub ratio: f64,
    pub format: String,
    pub neural_enabled: bool,
    pub lossy_savings: f64,
    pub quality: QualityLevel,
}

// WASM bindings

#[wasm_bindgen]
pub fn encode_stream_wasm(text: &str, chunk_size: usize) -> Result<Vec<u8>, JsValue> {
    encode_stream(text, chunk_size)
        .map_err(|e| JsValue::from_str(&format!("Encode error: {}", e)))
}

#[wasm_bindgen]
pub fn encode_stream_advanced_wasm(
    text: &str,
    chunk_size: usize,
    use_neural: bool,
    neural_weight: f32,
    quality_level: u8,
) -> Result<Vec<u8>, JsValue> {
    let options = CompressionOptions {
        chunk_size,
        use_neural,
        neural_weight,
        quality: QualityLevel::from_u8(quality_level),
        script: "devanagari".to_string(),
    };
    
    encode_stream_advanced(text, options)
        .map_err(|e| JsValue::from_str(&format!("Encode error: {}", e)))
}

#[wasm_bindgen]
pub fn decode_prefix_wasm(data: &[u8], upto: usize) -> Result<String, JsValue> {
    decode_prefix(data, upto)
        .map_err(|e| JsValue::from_str(&format!("Decode error: {}", e)))
}

#[wasm_bindgen]
pub fn decode_full_wasm(data: &[u8]) -> Result<String, JsValue> {
    decode_full(data)
        .map_err(|e| JsValue::from_str(&format!("Decode error: {}", e)))
}

#[wasm_bindgen]
pub fn get_compression_info(text: &str, chunk_size: usize) -> Result<String, JsValue> {
    let options = CompressionOptions {
        chunk_size,
        ..Default::default()
    };
    let stats = get_compression_stats(text, options)
        .map_err(|e| JsValue::from_str(&format!("Stats error: {}", e)))?;
    
    Ok(format!(
        "Original: {} bytes, Compressed: {} bytes, Ratio: {:.1}%, Format: {}, Neural: {}, Lossy: {:.1}%",
        stats.original_size,
        stats.compressed_size,
        stats.ratio,
        stats.format,
        stats.neural_enabled,
        stats.lossy_savings,
    ))
}

#[wasm_bindgen]
pub fn estimate_lossy_savings(text: &str, quality_level: u8) -> Result<String, JsValue> {
    let compressor = LossyCompressor::new(QualityLevel::from_u8(quality_level));
    let savings = compressor.estimate_savings(text);
    Ok(format!("{:.1}%", savings))
}