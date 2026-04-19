// src/chunk.rs  —  chunk encoding with variable-length token IDs
//
// KEY CHANGE: Token IDs are now encoded with a variable-length integer (VarInt)
// instead of a fixed 4 bytes.  After frequency-sorting the vocabulary the most
// common words get IDs 0-127 and each encodes in a single byte, giving real
// compression for repetitive text.
//
// VARINT SCHEME
//   0x00-0x7F          → 1 byte   (IDs 0-127)
//   0x80 b1            → 2 bytes  (IDs 128-16511)
//   0xC0 b1 b2 b3 b4   → 5 bytes  (IDs 16512+, rare)
//
// STREAM FORMAT (binary, magic "IC")
//   2 B   magic "IC"
//   1 B   flags  (bit 0: reserved; currently always 0x00)
//   4 B   vocab_count  (u32 LE)
//   for each vocab entry:
//     VarInt  token_id
//     2 B     text_byte_len  (u16 LE)
//     N B     text  (UTF-8)
//   4 B   chunk_count  (u32 LE)
//   for each chunk:
//     4 B   token_count   (u32 LE)
//     1 B   neural_encoded (0 or 1)
//     4 B   data_len       (u32 LE)
//     N B   data           (VarInt-encoded token IDs, no separator)

use crate::dict::MultiTierDict;
use crate::neural::NeuralPredictor;
use crate::tokenize::Token;
use crate::chunk_neural;
use anyhow::{Result, anyhow};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Chunk {
    pub data: Vec<u8>,
    pub token_count: usize,
    pub neural_encoded: bool,
}

#[derive(Debug)]
pub struct Stream {
    pub chunks: Vec<Chunk>,
    pub vocabulary: HashMap<u32, String>,
}

impl Stream {
    pub fn new() -> Self {
        Self { chunks: Vec::new(), vocabulary: HashMap::new() }
    }
}

// ---------------------------------------------------------------------------
// VarInt helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn write_varint(data: &mut Vec<u8>, value: u32) {
    if value < 0x80 {
        data.push(value as u8);
    } else if value < 0x4080 {           // 128..16511
        let adjusted = value - 0x80;     // 0..16511
        data.push(0x80 | (adjusted >> 8) as u8);
        data.push((adjusted & 0xFF) as u8);
    } else {
        data.push(0xC0);
        data.extend_from_slice(&value.to_le_bytes());
    }
}

#[inline]
pub fn read_varint(data: &[u8], pos: &mut usize) -> Option<u32> {
    if *pos >= data.len() { return None; }
    let b0 = data[*pos];
    *pos += 1;
    if b0 < 0x80 {
        Some(b0 as u32)
    } else if b0 < 0xC0 {
        if *pos >= data.len() { return None; }
        let b1 = data[*pos];
        *pos += 1;
        let adjusted = ((b0 & 0x3F) as u32) << 8 | b1 as u32;
        Some(adjusted + 0x80)
    } else {
        if *pos + 3 >= data.len() { return None; }
        let v = u32::from_le_bytes(data[*pos..*pos+4].try_into().ok()?);
        *pos += 4;
        Some(v)
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

pub fn encode_chunk(dict: &mut MultiTierDict, tokens: Vec<Token>) -> Result<Chunk> {
    let token_count = tokens.len();
    let mut data = Vec::with_capacity(4 + token_count); // optimistic 1 byte/id
    data.extend_from_slice(&(token_count as u32).to_le_bytes());
    for t in &tokens {
        write_varint(&mut data, t.id);
        dict.observe(t.id);
    }
    Ok(Chunk { data, token_count, neural_encoded: false })
}

pub fn encode_chunk_with_neural(
    dict: &mut MultiTierDict,
    tokens: Vec<Token>,
    neural: Option<&mut NeuralPredictor>,
    alpha: f32,
) -> Result<Chunk> {
    let token_count = tokens.len();
    if let Some(neural_model) = neural {
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        let data = chunk_neural::encode_chunk_neural(dict, ids, neural_model, alpha)?;
        Ok(Chunk { data, token_count, neural_encoded: true })
    } else {
        encode_chunk(dict, tokens)
    }
}

// ---------------------------------------------------------------------------
// Decoding  (tokens are concatenated directly — separator is encoded as a token)
// ---------------------------------------------------------------------------

pub fn decode_chunk(dict: &MultiTierDict, chunk: &Chunk) -> Result<String> {
    if chunk.data.len() < 4 { return Ok(String::new()); }
    let token_count = u32::from_le_bytes(chunk.data[0..4].try_into()?) as usize;
    let mut result = String::with_capacity(token_count * 4);
    let mut pos = 4usize;
    for _ in 0..token_count {
        match read_varint(&chunk.data, &mut pos) {
            Some(id) => {
                if let Some(text) = dict.decode_token(id) {
                    result.push_str(&text);
                    // Space is encoded as a real token in the vocabulary,
                    // so no separator is added here; the space token is " "
                }
            }
            None => break,
        }
    }
    Ok(result)
}

pub fn decode_chunk_with_neural(
    dict: &MultiTierDict,
    chunk: &Chunk,
    _neural: Option<&mut NeuralPredictor>,
    _alpha: f32,
) -> Result<String> {
    decode_chunk(dict, chunk)
}

// ---------------------------------------------------------------------------
// Binary stream I/O
// ---------------------------------------------------------------------------

pub fn write_stream(stream: &Stream) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(b"IC");
    out.push(0x00u8);  // flags byte (reserved)

    // Vocabulary
    out.extend_from_slice(&(stream.vocabulary.len() as u32).to_le_bytes());
    for (&id, text) in &stream.vocabulary {
        let tb = text.as_bytes();
        if tb.len() > u16::MAX as usize {
            return Err(anyhow!("Token text too long ({} B)", tb.len()));
        }
        write_varint(&mut out, id);
        out.extend_from_slice(&(tb.len() as u16).to_le_bytes());
        out.extend_from_slice(tb);
    }

    // Chunks
    out.extend_from_slice(&(stream.chunks.len() as u32).to_le_bytes());
    for c in &stream.chunks {
        out.extend_from_slice(&(c.token_count as u32).to_le_bytes());
        out.push(c.neural_encoded as u8);
        out.extend_from_slice(&(c.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&c.data);
    }

    Ok(out)
}

pub fn read_stream(data: &[u8]) -> Result<Stream> {
    // Skip magic bytes if present
    let start = if data.len() >= 2 && (&data[0..2] == b"IC" || &data[0..2] == b"IL") {
        2
    } else {
        0
    };
    let d = &data[start..];

    // Legacy JSON fallback
    if d.first() == Some(&b'{') {
        return read_stream_json_legacy(d);
    }

    read_stream_binary(d)
}

fn read_stream_binary(d: &[u8]) -> Result<Stream> {
    let mut pos = 0usize;

    macro_rules! need {
        ($n:expr) => {
            if pos + $n > d.len() {
                return Err(anyhow!("Stream truncated at pos {}", pos));
            }
        };
    }
    macro_rules! u32_le { () => {{
        need!(4);
        let v = u32::from_le_bytes(d[pos..pos+4].try_into()?);
        pos += 4; v
    }}; }
    macro_rules! u16_le { () => {{
        need!(2);
        let v = u16::from_le_bytes(d[pos..pos+2].try_into()?);
        pos += 2; v
    }}; }

    // Flags byte (skip)
    need!(1);
    pos += 1;

    // Vocabulary
    let vocab_count = u32_le!() as usize;
    let mut vocabulary = HashMap::with_capacity(vocab_count);
    for _ in 0..vocab_count {
        let id = read_varint(d, &mut pos)
            .ok_or_else(|| anyhow!("Truncated vocab id"))?;
        let tlen = u16_le!() as usize;
        need!(tlen);
        let text = String::from_utf8(d[pos..pos+tlen].to_vec())?;
        pos += tlen;
        vocabulary.insert(id, text);
    }

    // Chunks
    let chunk_count = u32_le!() as usize;
    let mut chunks = Vec::with_capacity(chunk_count);
    for _ in 0..chunk_count {
        let token_count    = u32_le!() as usize;
        need!(1);
        let neural_encoded = d[pos] != 0; pos += 1;
        let data_len       = u32_le!() as usize;
        need!(data_len);
        let data           = d[pos..pos+data_len].to_vec();
        pos += data_len;
        chunks.push(Chunk { data, token_count, neural_encoded });
    }

    Ok(Stream { chunks, vocabulary })
}

fn read_stream_json_legacy(d: &[u8]) -> Result<Stream> {
    #[derive(serde::Deserialize, Default)]
    struct LegacyChunk {
        data: Vec<u8>,
        token_count: usize,
        #[serde(default)] neural_encoded: bool,
    }
    #[derive(serde::Deserialize, Default)]
    struct LegacyStream {
        chunks: Vec<LegacyChunk>,
        #[serde(default)] vocabulary: HashMap<u32, String>,
    }
    let l: LegacyStream = serde_json::from_slice(d).unwrap_or_default();
    Ok(Stream {
        vocabulary: l.vocabulary,
        chunks: l.chunks.into_iter().map(|c| Chunk {
            data: c.data, token_count: c.token_count, neural_encoded: c.neural_encoded,
        }).collect(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (MultiTierDict, Vec<Token>) {
        let mut d = MultiTierDict::new();
        let tokens = vec![
            Token { id: 0, text: "hello".to_string() },
            Token { id: 1, text: " ".to_string() },
            Token { id: 2, text: "world".to_string() },
        ];
        for t in &tokens { d.add_token_with_id(t.id, t.text.clone()); }
        (d, tokens)
    }

    #[test]
    fn test_varint_roundtrip() {
        for &v in &[0u32, 1, 127, 128, 255, 1000, 16383, 16512, 99999, u32::MAX] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let mut pos = 0;
            let got = read_varint(&buf, &mut pos).unwrap();
            assert_eq!(got, v, "varint roundtrip failed for {}", v);
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn test_varint_sizes() {
        let mut buf = Vec::new();
        write_varint(&mut buf, 0);   assert_eq!(buf.len(), 1);  buf.clear();
        write_varint(&mut buf, 127); assert_eq!(buf.len(), 1);  buf.clear();
        write_varint(&mut buf, 128); assert_eq!(buf.len(), 2);  buf.clear();
        write_varint(&mut buf, 16511); assert_eq!(buf.len(), 2); buf.clear();
        write_varint(&mut buf, 16512); assert_eq!(buf.len(), 5); buf.clear();
    }

    #[test]
    fn test_encode_decode_exact() {
        let (mut d, t) = sample();
        let chunk = encode_chunk(&mut d, t).unwrap();
        let decoded = decode_chunk(&d, &chunk).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_stream_roundtrip() {
        let (mut enc, tokens) = sample();
        let chunk = encode_chunk(&mut enc, tokens).unwrap();
        let mut stream = Stream::new();
        stream.vocabulary = enc.get_vocabulary();
        stream.chunks.push(chunk);

        let bytes = write_stream(&stream).unwrap();
        assert_eq!(&bytes[0..2], b"IC");

        let restored = read_stream(&bytes).unwrap();
        let mut dec = MultiTierDict::new();
        dec.restore_vocabulary(restored.vocabulary);
        assert_eq!(decode_chunk(&dec, &restored.chunks[0]).unwrap(), "hello world");
    }

    #[test]
    fn test_compression_beats_raw_for_repetitive() {
        // Use multi-byte Indic chars (each = 3 UTF-8 bytes) but VarInt ID = 1 byte
        // → 3× compression on the token data.
        // "अब" × 800 = 1600 chars = 4800 UTF-8 bytes; stream ≈ 1640 bytes < 4800.
        let text = "अब".repeat(800);
        let mut d = MultiTierDict::new();
        d.add_token_with_id(0, "अ".to_string());
        d.add_token_with_id(1, "ब".to_string());
        let tokens: Vec<Token> = text.chars().enumerate().map(|(_i, c)| Token {
            id: if c == 'अ' { 0 } else { 1 },
            text: c.to_string(),
        }).collect();
        let chunk = encode_chunk(&mut d, tokens).unwrap();
        let mut stream = Stream::new();
        stream.vocabulary = d.get_vocabulary();
        stream.chunks.push(chunk);
        let compressed = write_stream(&stream).unwrap();
        assert!(compressed.len() < text.len(),
            "compressed ({}) should beat raw ({})", compressed.len(), text.len());
    }
}