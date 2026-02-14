// src/ppm.rs - PPM (Prediction by Partial Matching) stub
// This is a simplified stub - you can replace with full PPM implementation if needed

use anyhow::Result;

/// PPM context model
pub struct PPMModel {
    contexts: std::collections::HashMap<Vec<u32>, std::collections::HashMap<u32, u32>>,
}

impl PPMModel {
    pub fn new() -> Self {
        Self {
            contexts: std::collections::HashMap::new(),
        }
    }
    
    /// Update model with observed token
    pub fn update(&mut self, context: &[u32], token: u32) {
        let ctx = context.to_vec();
        self.contexts
            .entry(ctx)
            .or_insert_with(std::collections::HashMap::new)
            .entry(token)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }
    
    /// Get probability distribution for given context
    pub fn get_probabilities(&self, context: &[u32]) -> Vec<(u32, f32)> {
        if let Some(token_counts) = self.contexts.get(context) {
            let total: u32 = token_counts.values().sum();
            let total_f = total as f32;
            
            let mut probs: Vec<(u32, f32)> = token_counts
                .iter()
                .map(|(&token, &count)| (token, count as f32 / total_f))
                .collect();
            
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            probs
        } else {
            vec![]
        }
    }
}

impl Default for PPMModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode data using PPM
pub fn encode_ppm(tokens: &[u32]) -> Result<Vec<u8>> {
    // Simplified encoding - just store tokens
    let mut data = Vec::new();
    data.extend(&(tokens.len() as u32).to_le_bytes());
    for &token in tokens {
        data.extend(&token.to_le_bytes());
    }
    Ok(data)
}

/// Decode PPM-encoded data
pub fn decode_ppm(data: &[u8]) -> Result<Vec<u32>> {
    use std::convert::TryInto;
    
    if data.len() < 4 {
        return Ok(vec![]);
    }
    
    let count = u32::from_le_bytes(data[0..4].try_into()?) as usize;
    let mut tokens = Vec::with_capacity(count);
    
    let mut pos = 4;
    for _ in 0..count {
        if pos + 4 > data.len() {
            break;
        }
        let token = u32::from_le_bytes(data[pos..pos+4].try_into()?);
        tokens.push(token);
        pos += 4;
    }
    
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ppm_model() {
        let mut model = PPMModel::new();
        
        // Update with some tokens
        model.update(&[1, 2], 3);
        model.update(&[1, 2], 3);
        model.update(&[1, 2], 4);
        
        let probs = model.get_probabilities(&[1, 2]);
        
        // Token 3 should have higher probability (appeared 2x vs 1x)
        assert!(probs.len() > 0);
        assert_eq!(probs[0].0, 3);
    }
    
    #[test]
    fn test_encode_decode() {
        let tokens = vec![1, 2, 3, 4, 5];
        let encoded = encode_ppm(&tokens).unwrap();
        let decoded = decode_ppm(&encoded).unwrap();
        
        assert_eq!(tokens, decoded);
    }
}