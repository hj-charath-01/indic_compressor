// src/chunk_neural.rs - Neural-Enhanced Chunk Encoding
// Patent-worthy: Lightweight neural predictor integrated with PPM for hybrid compression

use crate::dict::MultiTierDict;
use crate::neural::NeuralPredictor;
use anyhow::Result;

/// Encode chunk with neural-hybrid prediction
/// 
/// This is the patent-worthy innovation: combines PPM statistical model
/// with lightweight neural network for improved token prediction
pub fn encode_chunk_neural(
    dict: &mut MultiTierDict,
    tokens: Vec<u32>,
    neural: &mut NeuralPredictor,
    alpha: f32, // Hybrid weight: 0.0 = pure PPM, 1.0 = pure neural
) -> Result<Vec<u8>> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    // Build frequency model (PPM-style)
    let mut ppm_model = HybridFrequencyModel::new();
    let mut context: Vec<u32> = Vec::new();
    
    // First pass: build frequency statistics
    for &token in &tokens {
        ppm_model.update(token, &context);
        context.push(token);
        if context.len() > 3 {
            context.remove(0);
        }
    }
    
    // Second pass: encode using hybrid predictions
    let mut encoded_data = Vec::new();
    context.clear();
    
    // For simplicity, we'll encode tokens with length-prefixed format
    // In production, use proper arithmetic/range coding
    encoded_data.extend(&(tokens.len() as u32).to_le_bytes());
    
    for &token in &tokens {
        // Get PPM frequencies
        let ppm_freqs = ppm_model.get_frequencies(&context);
        
        // Get neural predictions
        let neural_probs = neural.predict_token(&context, token);
        
        // Hybrid combination
        let hybrid_freqs = combine_predictions(&ppm_freqs, &neural_probs, alpha);
        
        // Find token rank in hybrid model
        let rank = hybrid_freqs.iter()
            .position(|&(t, _)| t == token)
            .unwrap_or(hybrid_freqs.len().saturating_sub(1));
        
        // Encode rank (smaller = more probable = fewer bits)
        encode_rank(&mut encoded_data, rank, hybrid_freqs.len());
        
        // Update context
        context.push(token);
        if context.len() > 3 {
            context.remove(0);
        }
    }
    
    Ok(encoded_data)
}

/// Hybrid frequency model combining PPM statistics
struct HybridFrequencyModel {
    unigrams: std::collections::HashMap<u32, u32>,
    bigrams: std::collections::HashMap<(u32, u32), u32>,
    trigrams: std::collections::HashMap<(u32, u32, u32), u32>,
}

impl HybridFrequencyModel {
    fn new() -> Self {
        Self {
            unigrams: std::collections::HashMap::new(),
            bigrams: std::collections::HashMap::new(),
            trigrams: std::collections::HashMap::new(),
        }
    }
    
    fn get_frequencies(&self, context: &[u32]) -> Vec<(u32, f32)> {
        // Get all seen tokens
        let mut freq_map: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        
        // Start with unigram frequencies
        for (&token, &count) in &self.unigrams {
            freq_map.insert(token, count as f32);
        }
        
        // Blend with bigram if available
        if context.len() >= 1 {
            let last = context[context.len() - 1];
            for (&(ctx, tok), &count) in &self.bigrams {
                if ctx == last {
                    *freq_map.entry(tok).or_insert(0.0) += count as f32 * 2.0;
                }
            }
        }
        
        // Blend with trigram if available
        if context.len() >= 2 {
            let last2 = context[context.len() - 2];
            let last1 = context[context.len() - 1];
            for (&(ctx2, ctx1, tok), &count) in &self.trigrams {
                if ctx2 == last2 && ctx1 == last1 {
                    *freq_map.entry(tok).or_insert(0.0) += count as f32 * 4.0;
                }
            }
        }
        
        // Convert to sorted vector
        let mut freqs: Vec<(u32, f32)> = freq_map.into_iter().collect();
        freqs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Add escape symbol
        freqs.push((u32::MAX, 1.0));
        
        freqs
    }
    
    fn update(&mut self, token: u32, context: &[u32]) {
        // Update unigram
        *self.unigrams.entry(token).or_insert(0) += 1;
        
        // Update bigram
        if context.len() >= 1 {
            let last = context[context.len() - 1];
            *self.bigrams.entry((last, token)).or_insert(0) += 1;
        }
        
        // Update trigram
        if context.len() >= 2 {
            let last2 = context[context.len() - 2];
            let last1 = context[context.len() - 1];
            *self.trigrams.entry((last2, last1, token)).or_insert(0) += 1;
        }
    }
}

/// Combine PPM frequencies with neural predictions
/// 
/// This is the core patent-worthy algorithm:
/// hybrid_freq = (1 - α) * freq_ppm + α * prob_neural * total
fn combine_predictions(
    ppm_freqs: &[(u32, f32)],
    neural_probs: &[(u32, f32)],
    alpha: f32,
) -> Vec<(u32, f32)> {
    let total: f32 = ppm_freqs.iter().map(|(_, f)| f).sum();
    let mut combined = std::collections::HashMap::new();
    
    // Add PPM contribution (weighted)
    for &(token, freq) in ppm_freqs {
        combined.insert(token, freq * (1.0 - alpha));
    }
    
    // Add neural contribution (weighted)
    for &(token, prob) in neural_probs {
        *combined.entry(token).or_insert(0.0) += prob * total * alpha;
    }
    
    // Convert back to vector
    let mut result: Vec<(u32, f32)> = combined.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    result
}

/// Encode rank using variable-length encoding
/// Lower ranks (more probable tokens) use fewer bytes
fn encode_rank(data: &mut Vec<u8>, rank: usize, vocab_size: usize) {
    // Use variable-length encoding:
    // - Rank 0-127: 1 byte
    // - Rank 128-16383: 2 bytes
    // - Rank 16384+: 3 bytes
    
    if rank < 128 {
        data.push(rank as u8);
    } else if rank < 16384 {
        data.push(0x80 | ((rank >> 8) as u8));
        data.push((rank & 0xFF) as u8);
    } else {
        let clamped = rank.min(vocab_size - 1);
        data.push(0xC0);
        data.extend(&(clamped as u16).to_le_bytes());
    }
}

/// Decode rank from variable-length encoding
#[allow(dead_code)]
fn decode_rank(data: &[u8], pos: &mut usize) -> Option<usize> {
    if *pos >= data.len() {
        return None;
    }
    
    let first = data[*pos];
    *pos += 1;
    
    if first < 0x80 {
        Some(first as usize)
    } else if first < 0xC0 {
        if *pos >= data.len() {
            return None;
        }
        let second = data[*pos];
        *pos += 1;
        Some((((first & 0x3F) as usize) << 8) | (second as usize))
    } else {
        if *pos + 1 >= data.len() {
            return None;
        }
        let low = data[*pos];
        let high = data[*pos + 1];
        *pos += 2;
        Some(((high as usize) << 8) | (low as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hybrid_encoding() {
        let mut dict = MultiTierDict::new();
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2];
        let mut neural = NeuralPredictor::new(8, 16, 256);
        
        let encoded = encode_chunk_neural(&mut dict, tokens.clone(), &mut neural, 0.3);
        assert!(encoded.is_ok());
        
        let encoded_bytes = encoded.unwrap();
        assert!(encoded_bytes.len() > 0);
        assert!(encoded_bytes.len() < tokens.len() * 4); // Should be compressed
    }
    
    #[test]
    fn test_frequency_blending() {
        let ppm = vec![(1, 10.0), (2, 5.0), (3, 2.0)];
        let neural = vec![(1, 0.5), (2, 0.3), (3, 0.2)];
        
        let combined = combine_predictions(&ppm, &neural, 0.5);
        
        // Token 1 should have highest combined frequency
        assert_eq!(combined[0].0, 1);
        assert!(combined[0].1 > combined[1].1);
    }
    
    #[test]
    fn test_rank_encoding() {
        let mut data = Vec::new();
        
        encode_rank(&mut data, 50, 1000);
        assert_eq!(data.len(), 1); // Small rank -> 1 byte
        
        data.clear();
        encode_rank(&mut data, 500, 1000);
        assert_eq!(data.len(), 2); // Medium rank -> 2 bytes
        
        data.clear();
        encode_rank(&mut data, 20000, 30000);
        assert_eq!(data.len(), 3); // Large rank -> 3 bytes
    }
}