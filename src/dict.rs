// src/dict.rs - Multi-tier dictionary for compression
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-tier dictionary for token compression
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MultiTierDict {
    /// Token to text mapping
    tokens: HashMap<u32, String>,
    /// Next token ID
    next_id: u32,
}

impl MultiTierDict {
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            next_id: 1,
        }
    }
    
    /// Observe a token (for frequency tracking)
    pub fn observe(&mut self, _token_id: u32) {
        // Update statistics (simplified)
    }
    
    /// Decode a token ID back to text
    pub fn decode_token(&self, token_id: u32) -> Option<String> {
        self.tokens.get(&token_id).cloned()
    }
    
    /// Add a new token
    pub fn add_token(&mut self, text: String) -> u32 {
        let id = self.next_id;
        self.tokens.insert(id, text);
        self.next_id += 1;
        id
    }
}

impl Default for MultiTierDict {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dict() {
        let mut dict = MultiTierDict::new();
        let id = dict.add_token("hello".to_string());
        
        assert_eq!(dict.decode_token(id), Some("hello".to_string()));
    }
}