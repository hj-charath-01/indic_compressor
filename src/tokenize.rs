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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenize_with_vocabulary() {
        let text = "hello world hello world";
        let tokens = tokenize(text);
        
        assert_eq!(tokens.len(), 4);
        
        // First "hello" and second "hello" should have SAME ID
        assert_eq!(tokens[0].id, tokens[2].id);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[2].text, "hello");
        
        // First "world" and second "world" should have SAME ID  
        assert_eq!(tokens[1].id, tokens[3].id);
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[3].text, "world");
        
        // "hello" and "world" should have DIFFERENT IDs
        assert_ne!(tokens[0].id, tokens[1].id);
    }
    
    #[test]
    fn test_tokenize_chars() {
        let text = "aabba";
        let tokens = tokenize_chars(text);
        
        assert_eq!(tokens.len(), 5);
        
        // All 'a's should have same ID
        assert_eq!(tokens[0].id, tokens[1].id);
        assert_eq!(tokens[0].id, tokens[4].id);
        
        // All 'b's should have same ID
        assert_eq!(tokens[2].id, tokens[3].id);
        
        // 'a' and 'b' should have different IDs
        assert_ne!(tokens[0].id, tokens[2].id);
    }
    
    #[test]
    fn test_repeated_patterns() {
        let text = "the cat and the dog and the bird";
        let tokens = tokenize(text);
        
        // Count unique token IDs
        let unique_ids: std::collections::HashSet<_> = 
            tokens.iter().map(|t| t.id).collect();
        
        // Should have only 5 unique words: the, cat, and, dog, bird
        assert_eq!(unique_ids.len(), 5);
        
        // But 8 total tokens (with repetitions)
        assert_eq!(tokens.len(), 8);
        
        // "the" appears 3 times with same ID
        let the_tokens: Vec<_> = tokens.iter()
            .filter(|t| t.text == "the")
            .collect();
        assert_eq!(the_tokens.len(), 3);
        assert_eq!(the_tokens[0].id, the_tokens[1].id);
        assert_eq!(the_tokens[0].id, the_tokens[2].id);
    }
}