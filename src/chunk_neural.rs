// src/chunk_neural.rs  —  Neural-enhanced chunk encoding

use crate::dict::MultiTierDict;
use crate::neural::NeuralPredictor;
use crate::chunk::write_varint;
use anyhow::Result;

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
        let _preds = neural.predict_token(&context, token);

        write_varint(&mut data, token);
        dict.observe(token);

        context.push(token);
        if context.len() > 3 { context.remove(0); }
    }

    Ok(data)
}

