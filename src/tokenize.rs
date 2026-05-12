// src/tokenize.rs - Proper tokenization with vocabulary
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A token with ID and text
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Token {
    /// Unique token ID (same word = same ID)
    pub id: u32,
    /// Original text
    pub text: String,
}

/// Tokenize text into tokens using a vocabulary
/// This ensures repeated words get the SAME token ID
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut vocabulary: HashMap<String, u32> = HashMap::new();
    let mut next_id = 0u32;
    let mut tokens = Vec::new();
    
    // Split by whitespace and assign consistent IDs
    for word in text.split_whitespace() {
        let word_str = word.to_string();
        
        // Get existing ID or create new one
        let token_id = *vocabulary.entry(word_str.clone()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        
        tokens.push(Token {
            id: token_id,
            text: word_str,
        });
    }
    
    tokens
}

/// Tokenize with character-level tokens (better for compression)
pub fn tokenize_chars(text: &str) -> Vec<Token> {
    let mut vocabulary: HashMap<char, u32> = HashMap::new();
    let mut next_id = 0u32;
    let mut tokens = Vec::new();
    
    for ch in text.chars() {
        let token_id = *vocabulary.entry(ch).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        
        tokens.push(Token {
            id: token_id,
            text: ch.to_string(),
        });
    }
    
    tokens
}
