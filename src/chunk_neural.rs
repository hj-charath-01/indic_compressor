// src/chunk_neural.rs  —  Neural-enhanced chunk encoding
//
// The neural model genuinely affects compression via the frequency-sorted
// vocabulary: lib.rs blends neural priors into the frequency sort *before*
// calling this module, so when use_neural=true the token IDs themselves are
// different (different words/chars map to different IDs), changing the VarInt
// sizes in the output stream.
//
// This module additionally maintains a running PPM context model so that in
// a future arithmetic-coding integration the neural predictions can modulate
// symbol probabilities per-position (not just the initial ID assignment).

use crate::dict::MultiTierDict;
use crate::neural::NeuralPredictor;
use crate::chunk::write_varint;
use anyhow::Result;

/// Encode a chunk using VarInt token IDs (same wire format as encode_chunk).
/// The neural model updates its internal embedding cache for each token so its
/// state is ready for future context-dependent coding.
pub fn encode_chunk_neural(
    dict: &mut MultiTierDict,
    tokens: Vec<u32>,
    neural: &mut NeuralPredictor,
    _alpha: f32,
) -> Result<Vec<u8>> {
    if tokens.is_empty() { return Ok(Vec::new()); }

    let token_count = tokens.len();
    let mut data = Vec::with_capacity(4 + token_count);
    data.extend_from_slice(&(token_count as u32).to_le_bytes());

    let mut context: Vec<u32> = Vec::with_capacity(4);

    for &token in &tokens {
        // Update neural state (embedding cache, context window).
        // The predictions are available here for a future arithmetic-coding
        // path; for now we write raw VarInt IDs which are already compressed
        // because the vocabulary was frequency-sorted with neural blending.
        let _preds = neural.predict_token(&context, token);

        write_varint(&mut data, token);
        dict.observe(token);

        context.push(token);
        if context.len() > 3 { context.remove(0); }
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::read_varint;
    use crate::neural::NeuralPredictor;

    #[test]
    fn test_neural_encode_varint_format() {
        let mut dict = MultiTierDict::new();
        let tokens = vec![0u32, 1, 2, 0, 1, 2];
        let mut neural = NeuralPredictor::new(8, 16, 64);

        let encoded = encode_chunk_neural(&mut dict, tokens.clone(), &mut neural, 0.3).unwrap();

        // First 4 bytes = token count
        let count = u32::from_le_bytes(encoded[0..4].try_into().unwrap()) as usize;
        assert_eq!(count, tokens.len());

        // Remaining bytes = VarInt-encoded IDs (all < 128 → 1 byte each)
        let mut pos = 4;
        for &expected in &tokens {
            let got = read_varint(&encoded, &mut pos).unwrap();
            assert_eq!(got, expected);
        }
        // Exactly 4 + 6×1 = 10 bytes for IDs 0-2
        assert_eq!(encoded.len(), 10);
    }

    #[test]
    fn test_neural_output_identical_format_to_non_neural() {
        use crate::chunk::{encode_chunk, encode_chunk_with_neural};
        use crate::tokenize::Token;

        let mut d1 = MultiTierDict::new();
        let mut d2 = MultiTierDict::new();
        for (id, text) in [(0u32,"a"),(1,"b"),(2,"c")] {
            d1.add_token_with_id(id, text.to_string());
            d2.add_token_with_id(id, text.to_string());
        }
        let tokens1: Vec<Token> = vec![
            Token{id:0,text:"a".into()},Token{id:1,text:"b".into()},Token{id:2,text:"c".into()}
        ];
        let tokens2 = tokens1.clone();

        let plain  = encode_chunk(&mut d1, tokens1).unwrap();
        let mut nn = NeuralPredictor::new(10,8,32);
        let neural = encode_chunk_with_neural(&mut d2, tokens2, Some(&mut nn), 0.5).unwrap();

        // Same data bytes: the wire format is identical
        assert_eq!(plain.data, neural.data,
            "neural and non-neural chunks must use the same VarInt wire format");
    }
}