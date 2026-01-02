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

    // Build PPM (over ranks) + rANS
    let mut ppm = PPMModel::new(2, vocab_size);
    let mut coder = DefaultAnsCoder::new();

    // We'll record the exact rank sequence that encoder passes to rANS (in decode order)
    let mut encoded_ranks: Vec<usize> = Vec::with_capacity(token_ids.len());

    // Prepare a list of (rank, model) in the reverse order the coder expects.
    let mut rank_model_pairs: Vec<(usize, Model)> = Vec::with_capacity(token_ids.len());

    // Reconstruct the same PPM progression to create models; do it in reverse as before:
    {
        let mut ppm_tmp = PPMModel::new(2, vocab_size);
        let mut prev_tmp: Vec<u32> = Vec::new();

        for i_rev in (0..token_ids.len()).rev() {
            let id = token_ids[i_rev];
            let rank = *id_to_rank.get(&id)
                .ok_or_else(|| anyhow!("id {} missing from rank mapping (encoder)", id))?;
            let rank_u32 = rank as u32;
            let features = features_vec[i_rev];

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

            // push the (symbol, model) pair in the order we will encode (reverse order)
            rank_model_pairs.push((rank, model));
            // record rank for debugging (this vector is in the same reverse order)
            encoded_ranks.push(rank);

            // update tmp ppm as before
            ppm_tmp.update(&prev_tmp, features, rank_u32);
            prev_tmp.push(rank_u32);
            if prev_tmp.len() > 2 { prev_tmp.remove(0); }
        }
    }

    // Now encode all pairs in the exact order recorded.
    {
        coder.encode_symbols(rank_model_pairs.into_iter())
            .map_err(|e| anyhow!("rANS encode_symbols failed: {:?}", e))?;
    }

    // finalize the coder (consumes coder) and get compressed words
    let words: Vec<u32> = coder.into_compressed()
        .map_err(|e| anyhow!("failed to finalize coder: {:?}", e))?;

    // serialize words -> big-endian bytes
    let mut payload_bytes: Vec<u8> = Vec::with_capacity(words.len() * 4);
    for &w in &words {
        payload_bytes.extend(&w.to_be_bytes());
    }

    // Debug print (optional) — shows rANS input sequence and payload words:
    println!("ENCODER: encoded_ranks (reverse-order) = {:?}", encoded_ranks);
    println!("ENCODER: payload words = {:?}", words);

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
    let mut id_to_rank: HashMap<u32, usize> = HashMap::new();
    for (r, &id) in rank_to_id.iter().enumerate() {
        id_to_rank.insert(id, r);
    }
    let vocab_size = rank_to_id.len(); // usize

    // 2) Deserialize rANS words
    if ch.payload.len() % 4 != 0 {
        return Err(anyhow!("payload length not a multiple of 4"));
    }
    let mut words = Vec::with_capacity(ch.payload.len() / 4);
    for i in (0..ch.payload.len()).step_by(4) {
        words.push(u32::from_be_bytes(ch.payload[i..i + 4].try_into().unwrap()));
    }

    // 3) Phase 1: rANS reverse decoding using the SAME PPM sequence (over ranks)
    let mut coder = DefaultAnsCoder::from_compressed(words)
        .map_err(|v| anyhow!("failed to create rANS decoder, leftover words: {:?}", v))?;
    let mut decoded_ids_rev: Vec<u32> = Vec::with_capacity(ch.token_count as usize);

    // Recreate the PPM in reverse order and decode using the matching models (ranks).
    let mut ppm_rev = PPMModel::new(2, vocab_size);
    let mut prev: Vec<u32> = Vec::new(); // store ranks as u32

    for i in (0..ch.token_count as usize).rev() {
        let feature_mask = ch.features[i];

        let (freqs, _total) = ppm_rev.get_freqs(&prev, feature_mask);
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

        // map rank -> id (dictionary id) for forward reconstruction
        let id = *rank_to_id.get(rank)
            .ok_or_else(|| anyhow!("rank {} out of range during decode", rank))?;
        decoded_ids_rev.push(id);

        // update the reverse-PPM with the decoded rank (so next reverse step
        // builds the same model sequence the encoder used)
        ppm_rev.update(&prev, feature_mask, rank as u32);
        prev.push(rank as u32);
        if prev.len() > 2 { prev.remove(0); }
    }

    // Reverse into forward order (ids)
    decoded_ids_rev.reverse();

    // 4) Phase 2: forward PPM reconstruction + text output (uses ids for lookup)
    let mut ppm_forward = PPMModel::new(2, dict.next_id() as usize);
    let mut prev_ids: Vec<u32> = Vec::new();
    let mut out = String::new();

    for (i, &id) in decoded_ids_rev.iter().enumerate() {
        let feature_mask = ch.features[i];
        ppm_forward.update(&prev_ids, feature_mask, id);

        if let Some(tok) = dict.lookup(id) {
            out.push_str(tok);
        } else {
            out.push_str("<UNK>");
        }

        prev_ids.push(id);
        if prev_ids.len() > 2 {
            prev_ids.remove(0);
        }
    }

    Ok(out)
}

/// Temporary verbose decoder for debugging. Prints internal state step-by-step.
pub fn decode_chunk_verbose(dict: &mut MultiTierDict, ch: &Chunk) -> Result<String> {
    println!("--- decode_chunk_verbose: token_count={} deltas={}", ch.token_count, ch.deltas.len());
    println!("deltas:");
    for (id, tok) in &ch.deltas {
        println!("  id={} tok=`{}`", id, tok);
    }

    // Apply deltas to dict (as real decoder does)
    for (id, tok) in &ch.deltas {
        dict.add_with_id(*id, tok);
    }

    // Build rank mapping
    let next_usize = dict.next_id() as usize;
    let mut rank_to_id: Vec<u32> = Vec::new();
    for id_usize in 0..next_usize {
        let id_u32 = id_usize as u32;
        if dict.lookup(id_u32).is_some() {
            rank_to_id.push(id_u32);
        }
    }
    println!("rank_to_id (len={}): {:?}", rank_to_id.len(), rank_to_id);

    let mut id_to_rank: HashMap<u32, usize> = HashMap::new();
    for (r, &id) in rank_to_id.iter().enumerate() {
        id_to_rank.insert(id, r);
    }

    // Deserialize payload -> words
    let mut words = Vec::with_capacity(ch.payload.len() / 4);
    for i in (0..ch.payload.len()).step_by(4) {
        let w = u32::from_be_bytes(ch.payload[i..i+4].try_into().unwrap());
        words.push(w);
    }
    println!("payload words (len={}): {:?}", words.len(), words);

    // Create coder
    let mut coder = DefaultAnsCoder::from_compressed(words.clone())
        .map_err(|v| anyhow!("failed to create rANS decoder: {:?}", v))?;

    // Reverse-phase decode (collect ranks as decoded)
    println!("-- reverse-phase decoding:");
    let mut ppm_rev = PPMModel::new(2, rank_to_id.len());
    let mut prev: Vec<u32> = Vec::new();
    let mut decoded_ids_rev: Vec<u32> = Vec::with_capacity(ch.token_count as usize);

    for i in (0..ch.token_count as usize).rev() {
        let feature_mask = ch.features[i];
        let (freqs, _total) = ppm_rev.get_freqs(&prev, feature_mask);
        let n = freqs.len();
        println!(" step rev i={} prev={:?} freqs.len={} first8={:?}", i, prev, n, &freqs.iter().take(8).cloned().collect::<Vec<_>>());
        let total_f: f64 = freqs.iter().map(|&v| v as f64).sum();
        let probs: Vec<f64> = if total_f > 0.0 {
            freqs.iter().map(|&v| (v as f64) / total_f).collect()
        } else {
            vec![1.0_f64 / (n as f64); n]
        };

        let model = Model::from_floating_point_probabilities_perfect(&probs)
            .or_else(|_| Model::from_floating_point_probabilities_perfect(&vec![1.0_f64 / (probs.len() as f64); probs.len()]))
            .map_err(|_| anyhow!("failed to build model (verbose decode)"))?;

        let rank = coder.decode_symbol(&model)
            .map_err(|e| anyhow!("rANS decode_symbol failed (verbose): {:?}", e))? as usize;
        let id = *rank_to_id.get(rank)
            .ok_or_else(|| anyhow!("rank {} out of range in verbose decode", rank))?;
        println!("  decoded rank={} -> id={}", rank, id);

        decoded_ids_rev.push(id);

        ppm_rev.update(&prev, feature_mask, rank as u32);
        prev.push(rank as u32);
        if prev.len() > 2 { prev.remove(0); }
    }

    println!("decoded_ids_rev (before reverse) = {:?}", decoded_ids_rev);
    decoded_ids_rev.reverse();
    println!("decoded_ids forward = {:?}", decoded_ids_rev);

    // Forward-phase: rebuild PPM over ids + output tokens
    println!("-- forward-phase reconstruct");
    let mut ppm_forward = PPMModel::new(2, dict.next_id() as usize);
    let mut prev_ids: Vec<u32> = Vec::new();
    let mut out = String::new();
    for (i, &id) in decoded_ids_rev.iter().enumerate() {
        let feature_mask = ch.features[i];
        ppm_forward.update(&prev_ids, feature_mask, id);
        println!(
            " forward i={} id={} tok=`{}` prev_ids={:?}",
            i,
            id,
            dict.lookup(id).map_or("<UNK>", |v| v.as_str()),
            prev_ids
        );
        if let Some(tok) = dict.lookup(id) {
            out.push_str(tok);
        } else {
            out.push_str("<UNK>");
        }
        prev_ids.push(id);
        if prev_ids.len() > 2 { prev_ids.remove(0); }
    }

    println!("final output: `{}`", out);
    Ok(out)
}
