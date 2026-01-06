// src/lib.rs - IMPROVED with size-aware compression

pub mod dict;
pub mod tokenize;
pub mod chunk;
pub mod ppm;

use anyhow::{Result, anyhow};
use wasm_bindgen::prelude::*;
use std::convert::TryInto;

/// Minimum text size (in bytes) before attempting compression
/// Below this threshold, data is stored uncompressed to avoid expansion
const COMPRESSION_THRESHOLD: usize = 500;

/// Encode an entire text into stream bytes with smart compression.
/// For small texts (<500 bytes), stores uncompressed to avoid expansion.
/// For larger texts, uses PPM + rANS compression with adaptive chunking.
///
/// `chunk_size` is number of tokens (grapheme clusters) per chunk.
pub fn encode_stream(text: &str, chunk_size: usize) -> Result<Vec<u8>> {
    let text_bytes = text.as_bytes().len();
    
    // For very small texts, compression overhead dominates - store uncompressed
    if text_bytes < COMPRESSION_THRESHOLD {
        return encode_uncompressed(text);
    }
    
    // Adaptive chunking: larger chunks for better compression on smaller texts
    let actual_chunk_size = if text_bytes < 2000 {
        200  // Large chunks for small-medium text
    } else if text_bytes < 10000 {
        100  // Medium chunks
    } else {
        chunk_size  // Use specified chunk size for large texts
    };
    
    let tokens = tokenize::tokenize(text);
    let mut dict = dict::MultiTierDict::new();
    let mut stream = chunk::Stream::new();

    for chunk_tokens in tokens.chunks(actual_chunk_size) {
        let ch = chunk::encode_chunk(&mut dict, chunk_tokens.to_vec())?;
        stream.chunks.push(ch);
    }

    chunk::write_stream(&stream)
}

/// Store text uncompressed with minimal overhead
fn encode_uncompressed(text: &str) -> Result<Vec<u8>> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    
    if len > u32::MAX as usize {
        return Err(anyhow!("Text too large for uncompressed format"));
    }
    
    // Format: "UC" (magic) + length (u32) + raw UTF-8 bytes
    let mut out = Vec::with_capacity(6 + len);
    out.extend_from_slice(b"UC");  // Magic: "UC" = Uncompressed
    out.extend_from_slice(&(len as u32).to_be_bytes());
    out.extend_from_slice(bytes);
    
    Ok(out)
}

/// Decode a full byte stream (handles both compressed and uncompressed formats)
pub fn decode_full(data: &[u8]) -> Result<String> {
    if data.len() < 2 {
        return Err(anyhow!("Data too short to be valid"));
    }
    
    // Check magic bytes to determine format
    let magic = &data[0..2];
    
    if magic == b"UC" {
        // Uncompressed format
        decode_uncompressed(data)
    } else if magic == b"IC" {
        // Compressed format (IC chunks)
        decode_compressed(data)
    } else {
        Err(anyhow!("Unknown format - magic bytes: {:?}", magic))
    }
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

/// Decode compressed format 
fn decode_compressed(data: &[u8]) -> Result<String> {
    let mut dict = dict::MultiTierDict::new();
    let stream = chunk::read_stream(data)?;
    let mut out = String::new();

    for ch in stream.chunks {
        let part = chunk::decode_chunk(&mut dict, &ch)?;
        out.push_str(&part);
    }
    
    Ok(out)
}

/// Decode only up to the first `upto` chunks (for partial/progressive decoding).
/// Note: Cannot partially decode uncompressed format.
pub fn decode_prefix(data: &[u8], upto: usize) -> Result<String> {
    if data.len() < 2 {
        return Err(anyhow!("Data too short to be valid"));
    }
    
    let magic = &data[0..2];
    
    if magic == b"UC" {
        // Uncompressed format - return full text (can't partially decode)
        decode_uncompressed(data)
    } else if magic == b"IC" {
        // Compressed format - decode specified number of chunks
        let mut dict = dict::MultiTierDict::new();
        let stream = chunk::read_stream(data)?;
        let mut out = String::new();

        for (i, ch) in stream.chunks.iter().enumerate() {
            if i >= upto { break; }
            let part = chunk::decode_chunk(&mut dict, &ch)?;
            out.push_str(&part);
        }
        
        Ok(out)
    } else {
        Err(anyhow!("Unknown format - magic bytes: {:?}", magic))
    }
}

/// Get compression statistics for a given text
pub fn get_compression_stats(text: &str, chunk_size: usize) -> Result<CompressionStats> {
    let original_size = text.as_bytes().len();
    let encoded = encode_stream(text, chunk_size)?;
    let compressed_size = encoded.len();
    
    let ratio = if original_size > 0 {
        (compressed_size as f64 / original_size as f64) * 100.0
    } else {
        0.0
    };
    
    let format = if encoded.len() >= 2 {
        if &encoded[0..2] == b"UC" {
            "uncompressed"
        } else {
            "compressed"
        }
    } else {
        "unknown"
    };
    
    Ok(CompressionStats {
        original_size,
        compressed_size,
        ratio,
        format: format.to_string(),
    })
}

#[derive(Debug)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub ratio: f64,
    pub format: String,
}

// WASM bindings

#[wasm_bindgen]
pub fn encode_stream_wasm(text: &str, chunk_size: usize) -> Result<Vec<u8>, JsValue> {
    encode_stream(text, chunk_size)
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
    let stats = get_compression_stats(text, chunk_size)
        .map_err(|e| JsValue::from_str(&format!("Stats error: {}", e)))?;
    
    Ok(format!(
        "Original: {} bytes, Compressed: {} bytes, Ratio: {:.1}%, Format: {}",
        stats.original_size,
        stats.compressed_size,
        stats.ratio,
        stats.format
    ))
}
