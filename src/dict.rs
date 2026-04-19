// src/dict.rs - Multi-tier dictionary for compression
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-tier dictionary for token compression
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct MultiTierDict {
    /// token_id -> text mapping
    tokens: HashMap<u32, String>,
    /// text -> token_id reverse mapping (not serialized; rebuilt on demand)
    #[serde(skip)]
    reverse: HashMap<String, u32>,
    /// Next token ID for auto-assignment
    next_id: u32,
}

impl MultiTierDict {
    pub fn new() -> Self {
        Self::default()
    }

    /// No-op frequency tracker (reserved for future PPM integration)
    pub fn observe(&mut self, _token_id: u32) {}

    /// Decode a token ID back to its original text
    pub fn decode_token(&self, token_id: u32) -> Option<String> {
        self.tokens.get(&token_id).cloned()
    }

    /// Add a new token with an auto-assigned ID, returning the ID.
    /// If the text already exists, returns the existing ID.
    pub fn add_token(&mut self, text: String) -> u32 {
        if let Some(&id) = self.reverse.get(&text) {
            return id;
        }
        let id = self.next_id;
        self.tokens.insert(id, text.clone());
        self.reverse.insert(text, id);
        self.next_id += 1;
        id
    }

    /// Register a token with a *specific* ID (used to sync tokenizer IDs into the dict).
    /// This is the critical method that makes encode/decode round-trip work.
    pub fn add_token_with_id(&mut self, id: u32, text: String) {
        self.reverse.insert(text.clone(), id);
        self.tokens.insert(id, text);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
    }

    /// Export the full vocabulary for serialisation into the compressed stream.
    pub fn get_vocabulary(&self) -> HashMap<u32, String> {
        self.tokens.clone()
    }

    /// Restore vocabulary from a deserialised stream so decode can map IDs → text.
    pub fn restore_vocabulary(&mut self, vocab: HashMap<u32, String>) {
        for (id, text) in vocab {
            self.reverse.insert(text.clone(), id);
            self.tokens.insert(id, text);
            if id >= self.next_id {
                self.next_id = id + 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dict_add_and_decode() {
        let mut dict = MultiTierDict::new();
        let id = dict.add_token("hello".to_string());
        assert_eq!(dict.decode_token(id), Some("hello".to_string()));
    }

    #[test]
    fn test_add_token_with_id() {
        let mut dict = MultiTierDict::new();
        dict.add_token_with_id(42, "world".to_string());
        assert_eq!(dict.decode_token(42), Some("world".to_string()));
    }

    #[test]
    fn test_vocabulary_roundtrip() {
        let mut dict = MultiTierDict::new();
        dict.add_token_with_id(0, "foo".to_string());
        dict.add_token_with_id(1, "bar".to_string());

        let vocab = dict.get_vocabulary();

        let mut dict2 = MultiTierDict::new();
        dict2.restore_vocabulary(vocab);
        assert_eq!(dict2.decode_token(0), Some("foo".to_string()));
        assert_eq!(dict2.decode_token(1), Some("bar".to_string()));
    }

    #[test]
    fn test_no_duplicate_ids() {
        let mut dict = MultiTierDict::new();
        let id1 = dict.add_token("hello".to_string());
        let id2 = dict.add_token("hello".to_string());
        assert_eq!(id1, id2);
    }
}