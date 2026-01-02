// src/chunk.rs
use crate::dict::MultiTierDict;
use crate::ppm::PPMModel;
use crate::tokenize::token_features;
use constriction::stream::stack::DefaultAnsCoder;
use constriction::stream::model::ContiguousCategoricalEntropyModel;
use constriction::stream::{Encode, Decode};
use anyhow::{Result, anyhow};
use std::io::{Cursor, Read, Write};
use std::convert::TryInto;
use std::collections::HashMap;

/// Choose a fixed precision for the entropy model (const generic).
const PRECISION: usize = 24;
type Model = ContiguousCategoricalEntropyModel<u32, Vec<u32>, PRECISION>;

/// Enable debug output (set to false for production)
const DEBUG: bool = true;

/// Chunk structure:
/// - token_count: number of tokens in the chunk
/// - deltas: newly-added token entries (id, token text)
/// - features: per-token feature mask (one u8 each)
/// - payload: rANS compressed words serialized as big-endian u32 bytes
pub struct Chunk {
    pub token_count: u32,
    pub deltas: Vec<(u32, String)>,
    pub features: Vec<u8>,
    pub payload: Vec<u8>,
}

pub struct Stream {
    pub chunks: Vec<Chunk>,
}
impl Stream {
    pub fn new() -> Self { Stream { chunks: Vec::new() } }
}

/// Binary framing for stream (per chunk):
/// - "IC" (2 bytes)
/// - token_count (u32 BE)
/// - delta_count (u16 BE)
/// - for each delta:
///     id (u32 BE)
///     token_len (u16 BE)
///     token bytes (UTF-8)
/// - features: token_count bytes (one u8 per token)
/// - payload_len (u32 BE)
/// - payload bytes (rANS compressed: big-endian u32 words)
pub fn write_stream(s: &Stream) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for ch in &s.chunks {
        out.write_all(b"IC")?;
        out.write_all(&ch.token_count.to_be_bytes())?;
        let del_count = ch.deltas.len() as u16;
        out.write_all(&del_count.to_be_bytes())?;
        for (id, tok) in &ch.deltas {
            let bs = tok.as_bytes();
            let len = bs.len() as u16;
            out.write_all(&id.to_be_bytes())?;
            out.write_all(&len.to_be_bytes())?;
            out.write_all(bs)?;
        }
        // features (exactly token_count bytes)
        if ch.features.len() != ch.token_count as usize {
            return Err(anyhow!("features length mismatch"));
        }
        out.write_all(&ch.features)?;
        out.write_all(&(ch.payload.len() as u32).to_be_bytes())?;
        out.write_all(&ch.payload)?;
    }
    Ok(out)
}

pub fn read_stream(data: &[u8]) -> Result<Stream> {
    let mut cur = Cursor::new(data);
    let mut chunks = Vec::new();
    loop {
        let mut magic = [0u8;2];
        match cur.read_exact(&mut magic) {
            Ok(()) => {},
            Err(_) => break, // EOF
        }
        if &magic != b"IC" {
            return Err(anyhow!("Bad magic header"));
        }
        let mut buf4 = [0u8;4];
        cur.read_exact(&mut buf4)?;
        let token_count = u32::from_be_bytes(buf4);
        let mut buf2 = [0u8;2];
        cur.read_exact(&mut buf2)?;
        let delta_count = u16::from_be_bytes(buf2);
        let mut deltas = Vec::with_capacity(delta_count as usize);
        for _ in 0..delta_count {
            cur.read_exact(&mut buf4)?;
            let id = u32::from_be_bytes(buf4);
            cur.read_exact(&mut buf2)?;
            let len = u16::from_be_bytes(buf2) as usize;
            let mut tb = vec![0u8; len];
            cur.read_exact(&mut tb)?;
            let tok = String::from_utf8(tb)?;
            deltas.push((id, tok));
        }
        // read features bytes (exactly token_count)
        let mut features = vec![0u8; token_count as usize];
        if token_count > 0 {
            cur.read_exact(&mut features)?;
        }
        cur.read_exact(&mut buf4)?;
        let payload_len = u32::from_be_bytes(buf4) as usize;
        let mut payload = vec![0u8; payload_len];
        cur.read_exact(&mut payload)?;
        chunks.push(Chunk { token_count, deltas, features, payload });
    }
    Ok(Stream { chunks })
}

/// Encode a single chunk given chunk tokens and dictionary state.
///
/// Encodes ranks in reverse order (so rANS decoding unwinds in reverse).
pub fn encode_chunk(dict: &mut MultiTierDict, chunk_tokens: Vec<String>) -> Result<Chunk> {
    // assign ids and collect deltas
    let mut token_ids: Vec<u32> = Vec::with_capacity(chunk_tokens.len());
    let mut deltas: Vec<(u32, String)> = Vec::new();
    let mut features_vec: Vec<u8> = Vec::with_capacity(chunk_tokens.len());

    for tok in &chunk_tokens {
        let (id, is_new) = dict.get_or_insert(tok);
        token_ids.push(id);
        if is_new {
            deltas.push((id, tok.clone()));
        }
        // compute and store feature mask (1 byte)
        let fm = token_features(tok);
        features_vec.push(fm);
    }

    // Build rank mapping (deterministic): iterate 0..dict.next_id() and collect present ids.
    let next_usize = dict.next_id() as usize;
    let mut rank_to_id: Vec<u32> = Vec::new();
    for id_usize in 0..next_usize {
        let id_u32 = id_usize as u32;
        if dict.lookup(id_u32).is_some() {
            rank_to_id.push(id_u32);
        }
    }
    if rank_to_id.is_empty() {
        return Err(anyhow!("vocabulary empty when encoding"));
    }
    let mut id_to_rank: HashMap<u32, usize> = HashMap::new();
    for (r, &id) in rank_to_id.iter().enumerate() {
        id_to_rank.insert(id, r);
    }
    let vocab_size = rank_to_id.len(); // usize

    // Debug print rank mapping
    if DEBUG {
        println!("ENCODER: vocab_size={}, dict.next_id()={}", vocab_size, dict.next_id());
        println!("ENCODER: rank_to_id={:?}", rank_to_id);
        println!("ENCODER: token_ids={:?}", token_ids);
        // Print first few tokens in dictionary
        println!("ENCODER: First 10 dict entries:");
        for i in 0..10.min(dict.next_id() as usize) {
            if let Some(tok) = dict.lookup(i as u32) {
                println!("  ID {}: '{}'", i, tok);
            }
        }
    }

    // Build rANS coder
    let mut coder = DefaultAnsCoder::new();

    // We'll record the exact rank sequence that encoder passes to rANS
    let mut encoded_ranks: Vec<usize> = Vec::with_capacity(token_ids.len());

    // NEW APPROACH: Encode in FORWARD order but push to rANS in REVERSE order
    // This way both encoder and decoder build PPM from past context
    
    // First pass: build models and collect (rank, model) pairs in forward order
    let mut rank_model_pairs: Vec<(usize, Model)> = Vec::with_capacity(token_ids.len());
    let mut ppm_tmp = PPMModel::new(2, vocab_size);
    let mut prev_tmp: Vec<u32> = Vec::new();

    for i in 0..token_ids.len() {
        let id = token_ids[i];
        let rank = *id_to_rank.get(&id)
            .ok_or_else(|| anyhow!("id {} missing from rank mapping (encoder)", id))?;
        let rank_u32 = rank as u32;
        let features = features_vec[i];

        let (freqs, _total) = ppm_tmp.get_freqs(&prev_tmp, features);
        let n = freqs.len();
        if n == 0 {
            return Err(anyhow!("empty frequency vector in encoder"));
        }

        let total_f: f64 = freqs.iter().map(|&v| v as f64).sum();
        let probs: Vec<f64> = if total_f > 0.0 {
            freqs.iter().map(|&v| (v as f64) / total_f).collect()
        } else {
            vec![1.0_f64 / (n as f64); n]
        };

        let model: Model = Model::from_floating_point_probabilities_perfect(&probs)
            .or_else(|_| {
                let uni = vec![1.0_f64 / (probs.len() as f64); probs.len()];
                Model::from_floating_point_probabilities_perfect(&uni)
            })
            .map_err(|_| anyhow!("failed to build entropy model (encoder)"))?;

        // Store (rank, model) pair
        rank_model_pairs.push((rank, model));
        encoded_ranks.push(rank);

        // Update PPM with the rank we just processed
        ppm_tmp.update(&prev_tmp, features, rank_u32);
        prev_tmp.push(rank_u32);
        if prev_tmp.len() > 2 { prev_tmp.remove(0); }
    }

    // Second pass: encode in REVERSE order for rANS
    // rANS is LIFO, so we push in reverse to pop in forward order
    for (rank, model) in rank_model_pairs.into_iter().rev() {
        coder.encode_symbol(rank, model)
            .map_err(|e| anyhow!("rANS encode_symbol failed: {:?}", e))?;
    }

    // finalize the coder (consumes coder) and get compressed words
    let words: Vec<u32> = coder.into_compressed()
        .map_err(|e| anyhow!("failed to finalize coder: {:?}", e))?;

    // serialize words -> big-endian bytes
    let mut payload_bytes: Vec<u8> = Vec::with_capacity(words.len() * 4);
    for &w in &words {
        payload_bytes.extend(&w.to_be_bytes());
    }

    // Debug print (optional)
    if DEBUG {
        println!("ENCODER: encoded_ranks (forward-order) = {:?}", encoded_ranks);
        println!("ENCODER: payload words = {:?}", words);
        println!("ENCODER: Note - symbols pushed to rANS in reverse order so they decode forward");
    }

    Ok(Chunk { token_count: token_ids.len() as u32, deltas, features: features_vec, payload: payload_bytes })
}

/// Decode a chunk using the dictionary (must apply deltas first).
///
/// Two-phase approach:
/// 1) rANS reverse-phase: decode ranks in reverse order using a PPM built in reverse (ppm_rev)
/// 2) forward-phase: reconstruct PPM forward and map ids -> tokens
pub fn decode_chunk(dict: &mut MultiTierDict, ch: &Chunk) -> Result<String> {
    // 1) Apply dictionary deltas
    for (id, tok) in &ch.deltas {
        dict.add_with_id(*id, tok);
    }

    // Build rank mapping (must match encoder): iterate 0..dict.next_id() and collect present ids.
    let next_usize = dict.next_id() as usize;
    let mut rank_to_id: Vec<u32> = Vec::new();
    for id_usize in 0..next_usize {
        let id_u32 = id_usize as u32;
        if dict.lookup(id_u32).is_some() {
            rank_to_id.push(id_u32);
        }
    }
    if rank_to_id.is_empty() {
        return Err(anyhow!("vocabulary empty when decoding"));
    }
    let vocab_size = rank_to_id.len(); // usize

    // Debug print rank mapping
    if DEBUG {
        println!("\nDECODER: vocab_size={}, dict.next_id()={}", vocab_size, dict.next_id());
        println!("DECODER: rank_to_id={:?}", rank_to_id);
        println!("DECODER: Deltas applied: {:?}", ch.deltas);
        // Print first few tokens in dictionary
        println!("DECODER: First 10 dict entries:");
        for i in 0..10.min(dict.next_id() as usize) {
            if let Some(tok) = dict.lookup(i as u32) {
                println!("  ID {}: '{}'", i, tok);
            }
        }
    }

    // 2) Deserialize rANS words
    if ch.payload.len() % 4 != 0 {
        return Err(anyhow!("payload length not a multiple of 4"));
    }
    let mut words = Vec::with_capacity(ch.payload.len() / 4);
    for i in (0..ch.payload.len()).step_by(4) {
        words.push(u32::from_be_bytes(ch.payload[i..i + 4].try_into().unwrap()));
    }

    // 3) Phase 1: rANS decoding
    // The encoder encoded in reverse order for rANS (LIFO stack)
    // rANS decodes in forward order: rank[0], rank[1], ..., rank[len-1]
    // We build PPM from past context (same as encoder did)
    let mut coder = DefaultAnsCoder::from_compressed(words)
        .map_err(|v| anyhow!("failed to create rANS decoder, leftover words: {:?}", v))?;
    let mut decoded_ids: Vec<u32> = Vec::with_capacity(ch.token_count as usize);
    let mut decoded_ranks: Vec<usize> = Vec::with_capacity(ch.token_count as usize);

    // Build PPM from past context (same as encoder)
    let mut ppm = PPMModel::new(2, vocab_size);
    let mut prev: Vec<u32> = Vec::new(); // store ranks as u32

    for i in 0..ch.token_count as usize {  // FORWARD iteration
        let feature_mask = ch.features[i];

        let (freqs, _total) = ppm.get_freqs(&prev, feature_mask);
        let n = freqs.len();
        if n == 0 {
            return Err(anyhow!("empty frequency vector in decoder"));
        }

        let total_f: f64 = freqs.iter().map(|&v| v as f64).sum();
        let probs: Vec<f64> = if total_f > 0.0 {
            freqs.iter().map(|&v| (v as f64) / total_f).collect()
        } else {
            vec![1.0_f64 / (n as f64); n]
        };

        let model = Model::from_floating_point_probabilities_perfect(&probs)
            .or_else(|_| {
                let uni = vec![1.0_f64 / (probs.len() as f64); probs.len()];
                Model::from_floating_point_probabilities_perfect(&uni)
            })
            .map_err(|_| anyhow!("failed to build entropy model (decoder)"))?;

        // decode to a rank (0..vocab_size-1)
        let rank = coder.decode_symbol(&model)
            .map_err(|e| anyhow!("rANS decode_symbol failed: {:?}", e))? as usize;

        // map rank -> id (dictionary id)
        let id = *rank_to_id.get(rank)
            .ok_or_else(|| anyhow!("rank {} out of range during decode", rank))?;
        decoded_ids.push(id);
        decoded_ranks.push(rank);

        // update the PPM with the decoded rank (building context for next decode)
        ppm.update(&prev, feature_mask, rank as u32);
        prev.push(rank as u32);
        if prev.len() > 2 { prev.remove(0); }
    }

    if DEBUG {
        println!("DECODER: decoded_ranks={:?}", decoded_ranks);
        println!("DECODER: decoded_ids={:?}", decoded_ids);
    }

    // 4) Phase 2: Convert IDs to tokens
    let mut out = String::new();

    for &id in decoded_ids.iter() {
        if let Some(tok) = dict.lookup(id) {
            out.push_str(tok);
        } else {
            out.push_str("<UNK>");
        }
    }

    Ok(out)
}