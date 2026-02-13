// src/tokenize.rs - Token definitions and tokenization
use serde::{Deserialize, Serialize};

/// A token with ID and text
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Token {
    /// Unique token ID
    pub id: u32,
    /// Original text
    pub text: String,
}

/// Tokenize text into tokens
pub fn tokenize(text: &str) -> Vec<Token> {
    // Simplified tokenization - split by whitespace
    // In production, use proper Unicode-aware tokenizer
    text.split_whitespace()
        .enumerate()
        .map(|(i, word)| Token {
            id: (i as u32 + 1), // IDs start from 1
            text: word.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenize() {
        let text = "hello world test";
        let tokens = tokenize(text);
        
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "test");
    }
}