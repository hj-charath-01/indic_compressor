// src/lib.rs

// Make modules public so examples and external callers can access them.
pub mod dict;
pub mod tokenize;
pub mod chunk;
pub mod ppm;

use anyhow::Result;
use wasm_bindgen::prelude::*;

/// High-level convenience: encode an entire text into the stream bytes (chunking).
/// `chunk_size` is number of tokens (grapheme clusters) per chunk.
pub fn encode_stream(text: &str, chunk_size: usize) -> Result<Vec<u8>> {
    let tokens = tokenize::tokenize(text);
    let mut dict = dict::MultiTierDict::new();
    let mut stream = chunk::Stream::new();

    for chunk in tokens.chunks(chunk_size) {
        let ch = chunk::encode_chunk(&mut dict, chunk.to_vec())?;
        stream.chunks.push(ch);
    }

    chunk::write_stream(&stream)
}

/// High-level convenience: decode a full byte stream produced by encode_stream.
pub fn decode_full(data: &[u8]) -> Result<String> {
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
pub fn decode_prefix(data: &[u8], upto: usize) -> Result<String> {
    let mut dict = dict::MultiTierDict::new();
    let stream = chunk::read_stream(data)?;
    let mut out = String::new();

    for (i, ch) in stream.chunks.iter().enumerate() {
        if i >= upto { break; }
        let part = chunk::decode_chunk(&mut dict, &ch)?;
        out.push_str(&part);
    }
    Ok(out)
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