// src/lib.rs  —  Indic ANS Compressor

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
use std::collections::HashMap;
use lossy::{LossyCompressor, QualityLevel, LossyMetadata};
use neural::load_pretrained_model;
use tokenize::Token;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CompressionOptions {
    pub chunk_size: usize,
    pub use_neural: bool,
    pub neural_weight: f32,
    pub quality: QualityLevel,
    pub script: String,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            chunk_size: 200,
            use_neural: false,
            neural_weight: 0.3,
            quality: QualityLevel::Lossless,
            script: "devanagari".to_string(),
        }
    }
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

// ---------------------------------------------------------------------------
// Normalisation helper
// ---------------------------------------------------------------------------

/// Collapse all runs of whitespace to a single ASCII space and strip leading/
/// trailing whitespace.  This is applied before the IC encoding path so that
/// `decode(encode(text)) == normalise(text)` always holds.
fn normalise_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Encoding  —  public entry points
// ---------------------------------------------------------------------------

pub fn encode_stream(text: &str, chunk_size: usize) -> Result<Vec<u8>> {
    encode_stream_advanced(text, CompressionOptions { chunk_size, ..Default::default() })
}

pub fn encode_stream_advanced(text: &str, options: CompressionOptions) -> Result<Vec<u8>> {
    // --- Step 0: normalise whitespace (applies to ALL encoding paths) ---
    let text = normalise_whitespace(text);
    let text = text.as_str();

    // --- Step 1: optional lossy pre-processing ---
    let (processed_text, lossy_meta) = if options.quality != QualityLevel::Lossless {
        let c = LossyCompressor::new(options.quality);
        let compressed = c.compress(text);
        if compressed.as_bytes().len() < text.as_bytes().len() {
            let meta = LossyMetadata::new(options.quality, text, &compressed, &options.script);
            (compressed, Some(meta))
        } else {
            (text.to_string(), None)
        }
    } else {
        (text.to_string(), None)
    };

    // --- Step 2: normalise whitespace for the IC path ---
    let canonical = normalise_whitespace(&processed_text);

    // --- Step 3: try to produce a compressed IC stream ---
    let ic_result = build_compressed_stream(&canonical, &options);

    // --- Step 4: build the raw UC bytes (always correct, might be larger) ---
    let uc_bytes = build_uncompressed_bytes(&processed_text);

    // --- Step 5: pick the smaller representation ---
    let use_ic = match &ic_result {
        Ok(ic) => ic.len() < uc_bytes.len(),
        Err(_)  => false,
    };

    let stream_body = if use_ic {
        ic_result.unwrap()
    } else {
        uc_bytes
    };

    // --- Step 6: prepend lossy metadata header if needed ---
    if let Some(meta) = &lossy_meta {
        let meta_bytes = meta.to_bytes();
        let body_magic = &stream_body[0..2];
        let outer_magic: &[u8] = match body_magic {
            b"IC" => b"IL",
            _     => b"UL",
        };
        let mut out = Vec::new();
        out.extend_from_slice(outer_magic);
        out.push(meta_bytes.len() as u8);
        out.extend_from_slice(&meta_bytes);
        out.extend_from_slice(&stream_body);
        return Ok(out);
    }

    Ok(stream_body)
}

/// Build an IC-format compressed stream for `text`.
/// Returns Err if the vocabulary is empty (empty input).
fn build_compressed_stream(text: &str, options: &CompressionOptions) -> Result<Vec<u8>> {
    if text.is_empty() {
        return Err(anyhow!("empty input"));
    }

    // --- Tokenise: words + spaces as separate tokens ---
    let raw_tokens = tokenise_with_spaces(text);
    if raw_tokens.is_empty() {
        return Err(anyhow!("no tokens"));
    }

    // --- Count corpus frequencies ---
    let mut freq: HashMap<u32, usize> = HashMap::new();
    let mut text_for_id: HashMap<u32, String> = HashMap::new();
    for t in &raw_tokens {
        *freq.entry(t.id).or_insert(0) += 1;
        text_for_id.entry(t.id).or_insert_with(|| t.text.clone());
    }

    // --- Score tokens: corpus frequency + optional neural prior ---
    let total_tokens = raw_tokens.len() as f64;
    let mut scores: Vec<(u32, f64)> = freq.iter()
        .map(|(&id, &cnt)| (id, cnt as f64))
        .collect();

    if options.use_neural {
        if let Some(neural) = load_pretrained_model(&options.script) {
            let alpha = options.neural_weight.clamp(0.0, 1.0) as f64;
            // b2 biases encode the Zipf-law prior trained for each script.
            // Range is approximately -8.0 (very rare) to +3.5 (very common).
            // We shift by +8 so all values are ≥ 0, then normalise to corpus-
            // count scale so the blend with corpus_freq is numerically stable.
            let b2_shift = 8.0f64;
            let max_shifted = 3.5 + b2_shift; // ≈ 11.5
            for (old_id, score) in &mut scores {
                if let Some(text) = text_for_id.get(old_id) {
                    if let Some(ch) = text.chars().next() {
                        let code = ch as u32;
                        let idx = (code % neural.output_size as u32) as usize;
                        let neural_score = if idx < neural.b2.len() {
                            (neural.b2[idx] as f64 + b2_shift).max(0.0)
                        } else {
                            0.0
                        };
                        // Scale neural score to the same order of magnitude as
                        // corpus frequency counts.
                        let neural_freq = neural_score / max_shifted * total_tokens;
                        *score = *score * (1.0 - alpha) + neural_freq * alpha;
                    }
                }
            }
        }
    }

    // --- Sort by score descending, assign new frequency-sorted IDs ---
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let id_remap: HashMap<u32, u32> = scores.iter().enumerate()
        .map(|(new_id, (old_id, _))| (*old_id, new_id as u32))
        .collect();

    // --- Build dictionary with remapped IDs ---
    let mut dict = dict::MultiTierDict::new();
    for (old_id, text) in &text_for_id {
        let new_id = id_remap[old_id];
        dict.add_token_with_id(new_id, text.clone());
    }

    // --- Remap tokens ---
    let remapped: Vec<Token> = raw_tokens.iter().map(|t| Token {
        id: id_remap[&t.id],
        text: t.text.clone(),
    }).collect();

    // --- Determine chunk size ---
    let chunk_size = if remapped.len() < 500 {
        remapped.len().max(1)
    } else {
        options.chunk_size.max(50)
    };

    // --- Load neural model for chunk-level encoding if requested ---
    let mut neural_model = if options.use_neural {
        load_pretrained_model(&options.script)
    } else {
        None
    };

    // --- Encode chunks ---
    let mut stream = chunk::Stream::new();
    for chunk_tokens in remapped.chunks(chunk_size) {
        let ch = if neural_model.is_some() {
            chunk::encode_chunk_with_neural(
                &mut dict,
                chunk_tokens.to_vec(),
                neural_model.as_mut(),
                options.neural_weight,
            )?
        } else {
            chunk::encode_chunk(&mut dict, chunk_tokens.to_vec())?
        };
        stream.chunks.push(ch);
    }
    stream.vocabulary = dict.get_vocabulary();

    chunk::write_stream(&stream)
}

/// Build raw UC (uncompressed) bytes for `text`.
fn build_uncompressed_bytes(text: &str) -> Vec<u8> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(6 + bytes.len());
    out.extend_from_slice(b"UC");
    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    out.extend_from_slice(bytes);
    out
}

// ---------------------------------------------------------------------------
// Tokeniser: words + spaces
// ---------------------------------------------------------------------------

fn tokenise_with_spaces(text: &str) -> Vec<Token> {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut next_id = 0u32;
    let mut tokens: Vec<Token> = Vec::new();
    let space_str = " ".to_string();

    for (i, word) in text.split_whitespace().enumerate() {
        if i > 0 {
            let space_id = *vocab.entry(space_str.clone()).or_insert_with(|| {
                let id = next_id; next_id += 1; id
            });
            tokens.push(Token { id: space_id, text: space_str.clone() });
        }
        let w = word.to_string();
        let word_id = *vocab.entry(w.clone()).or_insert_with(|| {
            let id = next_id; next_id += 1; id
        });
        tokens.push(Token { id: word_id, text: w });
    }
    tokens
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

pub fn decode_full(data: &[u8]) -> Result<String> {
    decode_full_with_options(data, None, 0.0)
}

pub fn decode_full_with_options(
    data: &[u8],
    neural_model: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    if data.len() < 2 { return Err(anyhow!("Data too short")); }
    match &data[0..2] {
        b"UC" => decode_uncompressed(data),
        b"UL" => decode_uncompressed_lossy(data),
        b"IC" => decode_compressed(data, neural_model, alpha),
        b"IL" => decode_lossy_wrapper(data, neural_model, alpha),
        m    => Err(anyhow!("Unknown magic: {:?}", m)),
    }
}

fn decode_uncompressed(data: &[u8]) -> Result<String> {
    if data.len() < 6 { return Err(anyhow!("Invalid UC: too short")); }
    let size = u32::from_be_bytes(data[2..6].try_into()?) as usize;
    if data.len() < 6 + size { return Err(anyhow!("Invalid UC: truncated")); }
    Ok(String::from_utf8(data[6..6+size].to_vec())?)
}

fn decode_uncompressed_lossy(data: &[u8]) -> Result<String> {
    if data.len() < 7 { return Err(anyhow!("Invalid UL: too short")); }
    let meta_len = data[2] as usize;
    if data.len() < 3 + meta_len + 4 { return Err(anyhow!("Invalid UL: truncated")); }
    let ds = 3 + meta_len;
    let size = u32::from_be_bytes(data[ds..ds+4].try_into()?) as usize;
    if data.len() < ds + 4 + size { return Err(anyhow!("Invalid UL: data truncated")); }
    Ok(String::from_utf8(data[ds+4..ds+4+size].to_vec())?)
}

fn decode_lossy_wrapper(
    data: &[u8],
    neural: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    if data.len() < 4 { return Err(anyhow!("Invalid IL: too short")); }
    let meta_len = data[2] as usize;
    if data.len() < 3 + meta_len { return Err(anyhow!("Invalid IL: truncated")); }
    let inner = &data[3 + meta_len..];
    decode_full_with_options(inner, neural, alpha)
}

fn decode_compressed(
    data: &[u8],
    mut neural: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    let mut d = dict::MultiTierDict::new();
    let stream = chunk::read_stream(data)?;
    d.restore_vocabulary(stream.vocabulary.clone());

    let mut out = String::new();
    for ch in &stream.chunks {
        let part = chunk::decode_chunk_with_neural(&d, ch, neural.as_deref_mut(), alpha)?;
        out.push_str(&part);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Prefix / partial decoding
// ---------------------------------------------------------------------------

pub fn decode_prefix(data: &[u8], upto: usize) -> Result<String> {
    decode_prefix_with_options(data, upto, None, 0.0)
}

pub fn decode_prefix_with_options(
    data: &[u8],
    upto: usize,
    neural: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    if data.len() < 2 { return Err(anyhow!("Data too short")); }
    match &data[0..2] {
        b"UC" | b"UL" => decode_full_with_options(data, neural, alpha),
        b"IC" => decode_compressed_prefix(data, upto, neural, alpha),
        b"IL" => {
            if data.len() < 4 { return Err(anyhow!("Invalid IL")); }
            let meta_len = data[2] as usize;
            decode_compressed_prefix(&data[3+meta_len..], upto, neural, alpha)
        }
        m => Err(anyhow!("Unknown format: {:?}", m)),
    }
}

fn decode_compressed_prefix(
    data: &[u8],
    upto: usize,
    mut neural: Option<&mut neural::NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    let mut d = dict::MultiTierDict::new();
    let stream = chunk::read_stream(data)?;
    d.restore_vocabulary(stream.vocabulary.clone());

    let mut out = String::new();
    for (i, ch) in stream.chunks.iter().enumerate() {
        if i >= upto { break; }
        out.push_str(&chunk::decode_chunk_with_neural(&d, ch, neural.as_deref_mut(), alpha)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

pub fn get_compression_stats(text: &str, options: CompressionOptions) -> Result<CompressionStats> {
    let original_size = text.as_bytes().len();
    let encoded = encode_stream_advanced(text, options.clone())?;
    let compressed_size = encoded.len();
    let ratio = if original_size > 0 { compressed_size as f64 / original_size as f64 * 100.0 } else { 0.0 };
    let format = if encoded.len() >= 2 {
        match &encoded[0..2] {
            b"UC" => "uncompressed", b"UL" => "uncompressed-lossy",
            b"IC" => "compressed",   b"IL" => "compressed-lossy",
            _ => "unknown",
        }
    } else { "unknown" };
    let lossy_savings = if options.quality != QualityLevel::Lossless {
        LossyCompressor::new(options.quality).estimate_savings(text)
    } else { 0.0 };
    Ok(CompressionStats {
        original_size, compressed_size, ratio,
        format: format.to_string(),
        neural_enabled: options.use_neural,
        lossy_savings, quality: options.quality,
    })
}

// ---------------------------------------------------------------------------
// WASM bindings
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub fn encode_stream_wasm(text: &str, chunk_size: usize) -> Result<Vec<u8>, JsValue> {
    encode_stream(text, chunk_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn encode_stream_advanced_wasm(
    text: &str,
    chunk_size: usize,
    use_neural: bool,
    neural_weight: f32,
    quality_level: u8,
) -> Result<Vec<u8>, JsValue> {
    encode_stream_advanced(text, CompressionOptions {
        chunk_size, use_neural, neural_weight,
        quality: QualityLevel::from_u8(quality_level),
        script: "devanagari".to_string(),
    }).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn decode_prefix_wasm(data: &[u8], upto: usize) -> Result<String, JsValue> {
    decode_prefix(data, upto)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn decode_full_wasm(data: &[u8]) -> Result<String, JsValue> {
    decode_full(data)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn get_compression_info(text: &str, chunk_size: usize) -> Result<String, JsValue> {
    let stats = get_compression_stats(text, CompressionOptions { chunk_size, ..Default::default() })
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(format!(
        "Original: {} bytes, Compressed: {} bytes, Ratio: {:.1}%, Format: {}, Neural: {}, Lossy: {:.1}%",
        stats.original_size, stats.compressed_size, stats.ratio,
        stats.format, stats.neural_enabled, stats.lossy_savings))
}

#[wasm_bindgen]
pub fn estimate_lossy_savings(text: &str, quality_level: u8) -> Result<String, JsValue> {
    Ok(format!("{:.1}%", LossyCompressor::new(QualityLevel::from_u8(quality_level)).estimate_savings(text)))
}

