// src/chunk.rs - Chunk encoding with neural integration
// This file provides the interface between lib.rs and the actual encoding implementations

use crate::dict::MultiTierDict;
use crate::neural::NeuralPredictor;
use crate::tokenize::Token;
use crate::chunk_neural;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// A compressed chunk of tokens
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Chunk {
    /// Compressed data for this chunk
    pub data: Vec<u8>,
    /// Number of original tokens
    pub token_count: usize,
    /// Whether neural encoding was used
    pub neural_encoded: bool,
}

/// Stream of compressed chunks
#[derive(Serialize, Deserialize, Debug)]
pub struct Stream {
    pub chunks: Vec<Chunk>,
}

impl Stream {
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
        }
    }
}

/// Encode a chunk WITHOUT neural prediction (standard PPM)
pub fn encode_chunk(dict: &mut MultiTierDict, tokens: Vec<Token>) -> Result<Chunk> {
    // Convert tokens to u32 IDs
    let token_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
    let token_count = token_ids.len();
    
    // For now, simple encoding (in production, use proper PPM + arithmetic coding)
    let mut data = Vec::new();
    data.extend(&(token_count as u32).to_le_bytes());
    
    for id in token_ids {
        data.extend(&id.to_le_bytes());
        dict.observe(id); // Update dictionary
    }
    
    Ok(Chunk {
        data,
        token_count,
        neural_encoded: false,
    })
}

/// Encode a chunk WITH neural-hybrid prediction
/// This is the key function that integrates neural compression!
pub fn encode_chunk_with_neural(
    dict: &mut MultiTierDict,
    tokens: Vec<Token>,
    neural: Option<&mut NeuralPredictor>,
    alpha: f32,
) -> Result<Chunk> {
    let token_count = tokens.len();
    
    // If we have a neural model, use hybrid encoding
    if let Some(neural_model) = neural {
        // Convert tokens to u32 IDs
        let token_ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        
        // Use the neural-hybrid encoder from chunk_neural.rs
        let data = chunk_neural::encode_chunk_neural(
            dict,
            token_ids,
            neural_model,
            alpha,
        )?;
        
        Ok(Chunk {
            data,
            token_count,
            neural_encoded: true,
        })
    } else {
        // No neural model, fall back to standard encoding
        encode_chunk(dict, tokens)
    }
}

/// Decode a chunk WITHOUT neural prediction
pub fn decode_chunk(dict: &mut MultiTierDict, chunk: &Chunk) -> Result<String> {
    if chunk.data.len() < 4 {
        return Ok(String::new());
    }
    
    let token_count = u32::from_le_bytes(chunk.data[0..4].try_into()?) as usize;
    let mut pos = 4;
    let mut result = String::new();
    
    for _ in 0..token_count {
        if pos + 4 > chunk.data.len() {
            break;
        }
        
        let token_id = u32::from_le_bytes(chunk.data[pos..pos+4].try_into()?);
        pos += 4;
        
        // Convert token ID back to text (simplified)
        if let Some(text) = dict.decode_token(token_id) {
            result.push_str(&text);
        }
    }
    
    Ok(result)
}

/// Decode a chunk WITH neural support
pub fn decode_chunk_with_neural(
    dict: &mut MultiTierDict,
    chunk: &Chunk,
    neural: Option<&mut NeuralPredictor>,
    alpha: f32,
) -> Result<String> {
    if chunk.neural_encoded && neural.is_some() {
        // Decode using neural-aware decoder
        decode_chunk_neural_aware(dict, chunk, neural.unwrap(), alpha)
    } else {
        // Standard decoding
        decode_chunk(dict, chunk)
    }
}

/// Neural-aware decoder (mirrors the encoding process)
fn decode_chunk_neural_aware(
    dict: &mut MultiTierDict,
    chunk: &Chunk,
    neural: &mut NeuralPredictor,
    alpha: f32,
) -> Result<String> {
    // This would implement the inverse of encode_chunk_neural
    // For now, use simplified decoding
    
    let data = &chunk.data;
    if data.len() < 4 {
        return Ok(String::new());
    }
    
    let token_count = u32::from_le_bytes(data[0..4].try_into()?) as usize;
    let mut pos = 4;
    let mut result = String::new();
    let mut context: Vec<u32> = Vec::new();
    
    for _ in 0..token_count {
        // Decode rank
        let rank = decode_rank(data, &mut pos)?;
        
        // Rebuild hybrid predictions to find token from rank
        let ppm_freqs = get_ppm_frequencies(dict, &context);
        let neural_probs = neural.predict_token(&context, 0); // dummy target
        let hybrid_freqs = combine_predictions_for_decode(&ppm_freqs, &neural_probs, alpha);
        
        // Get token at this rank
        let token_id = hybrid_freqs.get(rank).map(|(t, _)| *t).unwrap_or(0);
        
        // Convert to text
        if let Some(text) = dict.decode_token(token_id) {
            result.push_str(&text);
        }
        
        // Update context
        context.push(token_id);
        if context.len() > 3 {
            context.remove(0);
        }
    }
    
    Ok(result)
}

/// Helper function to decode rank
fn decode_rank(data: &[u8], pos: &mut usize) -> Result<usize> {
    if *pos >= data.len() {
        return Ok(0);
    }
    
    let first = data[*pos];
    *pos += 1;
    
    if first < 0x80 {
        Ok(first as usize)
    } else if first < 0xC0 {
        if *pos >= data.len() {
            return Ok(0);
        }
        let second = data[*pos];
        *pos += 1;
        Ok((((first & 0x3F) as usize) << 8) | (second as usize))
    } else {
        if *pos + 1 >= data.len() {
            return Ok(0);
        }
        let low = data[*pos];
        let high = data[*pos + 1];
        *pos += 2;
        Ok(((high as usize) << 8) | (low as usize))
    }
}

/// Get PPM frequencies (simplified)
fn get_ppm_frequencies(_dict: &MultiTierDict, _context: &[u32]) -> Vec<(u32, f32)> {
    // Placeholder - would use actual PPM model
    vec![(0, 1.0)]
}

/// Combine predictions for decoding
fn combine_predictions_for_decode(
    ppm_freqs: &[(u32, f32)],
    neural_probs: &[(u32, f32)],
    alpha: f32,
) -> Vec<(u32, f32)> {
    let total: f32 = ppm_freqs.iter().map(|(_, f)| f).sum();
    let mut combined = std::collections::HashMap::new();
    
    for &(token, freq) in ppm_freqs {
        combined.insert(token, freq * (1.0 - alpha));
    }
    
    for &(token, prob) in neural_probs {
        *combined.entry(token).or_insert(0.0) += prob * total * alpha;
    }
    
    let mut result: Vec<(u32, f32)> = combined.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    result
}

/// Write stream to bytes
pub fn write_stream(stream: &Stream) -> Result<Vec<u8>> {
    let json = serde_json::to_string(stream)?;
    let mut bytes = b"IC".to_vec(); // "IC" = Indic Compressed
    bytes.extend(json.as_bytes());
    Ok(bytes)
}

/// Read stream from bytes
pub fn read_stream(data: &[u8]) -> Result<Stream> {
    if data.len() < 2 {
        anyhow::bail!("Data too short");
    }
    
    // Skip magic bytes if present
    let json_start = if &data[0..2] == b"IC" || &data[0..2] == b"IL" {
        2
    } else {
        0
    };
    
    let stream: Stream = serde_json::from_slice(&data[json_start..])?;
    Ok(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunk_encoding() {
        let mut dict = MultiTierDict::new();
        let tokens = vec![
            Token { id: 1, text: "hello".to_string() },
            Token { id: 2, text: "world".to_string() },
        ];
        
        let chunk = encode_chunk(&mut dict, tokens).unwrap();
        assert!(chunk.data.len() > 0);
        assert_eq!(chunk.token_count, 2);
        assert!(!chunk.neural_encoded);
    }
    
    #[test]
    fn test_neural_chunk_encoding() {
        let mut dict = MultiTierDict::new();
        let mut neural = NeuralPredictor::new(100, 8, 32);
        let tokens = vec![
            Token { id: 1, text: "hello".to_string() },
            Token { id: 2, text: "world".to_string() },
        ];
        
        let chunk = encode_chunk_with_neural(&mut dict, tokens, Some(&mut neural), 0.5).unwrap();
        assert!(chunk.data.len() > 0);
        assert_eq!(chunk.token_count, 2);
        assert!(chunk.neural_encoded);
    }
}