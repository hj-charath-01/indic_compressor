// src/neural.rs - Lightweight Neural Predictor for Indic Text Compression

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tiny neural network for next-token prediction.
/// Architecture: (2 × embedding + 8 features) → hidden (ReLU) → output (softmax).
#[derive(Serialize, Deserialize, Clone)]
pub struct NeuralPredictor {
    pub input_size: usize,   // embedding_dim * 2 + 8
    pub hidden_size: usize,
    pub output_size: usize,

    pub w1: Vec<f32>, // (input_size × hidden_size)
    pub b1: Vec<f32>, // hidden_size
    pub w2: Vec<f32>, // (hidden_size × output_size)
    pub b2: Vec<f32>, // output_size

    pub embeddings: HashMap<u32, Vec<f32>>,
    pub embedding_dim: usize,
}

impl NeuralPredictor {
    /// Create a new predictor with Xavier-initialised weights.
    pub fn new(vocab_size: usize, embedding_dim: usize, hidden_size: usize) -> Self {
        let input_size = embedding_dim * 2 + 8; // 2 context embeddings + 8 feature bits
        let output_size = vocab_size;

        let w1 = Self::xavier_init(input_size, hidden_size);
        let b1 = vec![0.0f32; hidden_size];
        let w2 = Self::xavier_init(hidden_size, output_size);
        let b2 = vec![0.0f32; output_size];

        Self {
            input_size,
            hidden_size,
            output_size,
            w1,
            b1,
            w2,
            b2,
            embeddings: HashMap::new(),
            embedding_dim,
        }
    }

    fn xavier_init(in_size: usize, out_size: usize) -> Vec<f32> {
        let limit = (6.0_f32 / (in_size + out_size) as f32).sqrt();
        (0..in_size * out_size)
            .map(|i| {
                let x = ((i.wrapping_mul(2_654_435_761)) % 2_147_483_647) as f32
                    / 2_147_483_647.0;
                (x * 2.0 - 1.0) * limit
            })
            .collect()
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Get (or lazily create) the embedding for a token.
    fn get_embedding(&mut self, token_id: u32) -> Vec<f32> {
        if let Some(emb) = self.embeddings.get(&token_id) {
            return emb.clone();
        }
        let mut emb = vec![0.0f32; self.embedding_dim];
        let id_f = token_id as f32;
        for i in 0..self.embedding_dim {
            emb[i] = ((id_f * (i as f32 + 1.0)).sin() * 0.1).tanh();
        }
        self.embeddings.insert(token_id, emb.clone());
        emb
    }

    /// Full forward pass: returns a probability distribution over the vocabulary.
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Hidden layer with ReLU
        let mut hidden = vec![0.0f32; self.hidden_size];
        let in_len = input.len().min(self.input_size);
        for h in 0..self.hidden_size {
            let mut s = self.b1[h];
            for i in 0..in_len {
                s += input[i] * self.w1[i * self.hidden_size + h];
            }
            hidden[h] = s.max(0.0);
        }

        // Output layer
        let mut output = vec![0.0f32; self.output_size];
        for o in 0..self.output_size {
            let mut s = self.b2[o];
            for h in 0..self.hidden_size {
                s += hidden[h] * self.w2[h * self.output_size + o];
            }
            output[o] = s;
        }

        Self::softmax_inplace(&mut output);
        output
    }

    fn softmax_inplace(logits: &mut [f32]) {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in logits.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        if sum > 0.0 {
            for x in logits.iter_mut() {
                *x /= sum;
            }
        }
    }

    /// Predict next-token probabilities given a context window.
    /// Returns the top-32 (token_id, probability) pairs sorted by descending probability.
    pub fn predict_token(&mut self, context: &[u32], _target: u32) -> Vec<(u32, f32)> {
        // Build two-token context embedding (padded with zeros if context is short)
        let emb_prev2 = if context.len() >= 2 {
            self.get_embedding(context[context.len() - 2])
        } else {
            vec![0.0f32; self.embedding_dim]
        };
        let emb_prev1 = if !context.is_empty() {
            self.get_embedding(context[context.len() - 1])
        } else {
            vec![0.0f32; self.embedding_dim]
        };

        let features = self.extract_features(context);

        // Concatenate: [emb_prev2 | emb_prev1 | features]  → length == input_size
        let mut input = Vec::with_capacity(self.input_size);
        input.extend_from_slice(&emb_prev2);
        input.extend_from_slice(&emb_prev1);
        input.extend_from_slice(&features);

        let output = self.forward(&input);

        let mut token_probs: Vec<(u32, f32)> = output
            .iter()
            .enumerate()
            .map(|(idx, &p)| (idx as u32, p))
            .collect();
        token_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        token_probs.truncate(32);
        token_probs
    }

    /// Optional online learning hook (no-op for now).
    pub fn observe(&mut self, _context: &[u32], _token: u32) {}

    fn extract_features(&self, context: &[u32]) -> Vec<f32> {
        let mut features = vec![0.0f32; 8];
        features[0] = if context.len() >= 1 { 1.0 } else { 0.0 };
        features[1] = if context.len() >= 2 { 1.0 } else { 0.0 };

        if let Some(&last) = context.last() {
            features[2] = if last < 128 { 1.0 } else { 0.0 };
            features[3] = if last >= 2304 && last < 2432 { 1.0 } else { 0.0 };

            if context.len() >= 2 {
                let prev = context[context.len() - 2];
                features[4] = if last == prev { 1.0 } else { 0.0 };
            }

            features[6] = (context.len() as f32) / 100.0;
            features[7] = (last % 10) as f32 / 10.0;
        }

        features
    }
}

// ---------------------------------------------------------------------------
// HybridPredictor
// ---------------------------------------------------------------------------

pub struct HybridPredictor {
    neural: Option<NeuralPredictor>,
    neural_weight: f32,
}

impl HybridPredictor {
    pub fn new(neural: Option<NeuralPredictor>, neural_weight: f32) -> Self {
        Self {
            neural,
            neural_weight: neural_weight.clamp(0.0, 1.0),
        }
    }

    pub fn combine_predictions(
        &mut self,
        ppm_probs: &[f32],
        context: &[u32],
        _features: u8,
    ) -> Vec<f32> {
        if let Some(ref mut neural) = self.neural {
            // Build input using the public predict pathway
            let neural_pairs = neural.predict_token(context, 0);
            let mut neural_probs = vec![0.0f32; ppm_probs.len()];
            for (id, p) in neural_pairs {
                if (id as usize) < neural_probs.len() {
                    neural_probs[id as usize] = p;
                }
            }

            let ppm_w = 1.0 - self.neural_weight;
            let combined_len = ppm_probs.len().min(neural_probs.len());
            let mut combined = vec![0.0f32; combined_len];
            for i in 0..combined_len {
                combined[i] = ppm_w * ppm_probs[i] + self.neural_weight * neural_probs[i];
            }
            let sum: f32 = combined.iter().sum();
            if sum > 0.0 {
                for p in &mut combined {
                    *p /= sum;
                }
            }
            combined
        } else {
            ppm_probs.to_vec()
        }
    }
}

// ---------------------------------------------------------------------------
// Script detection
// ---------------------------------------------------------------------------

const UNICODE_RANGES: &[(&str, u32, u32)] = &[
    ("devanagari", 0x0900, 0x097F),
    ("bengali",    0x0980, 0x09FF),
    ("gurmukhi",   0x0A00, 0x0A7F),
    ("gujarati",   0x0A80, 0x0AFF),
    ("oriya",      0x0B00, 0x0B7F),
    ("tamil",      0x0B80, 0x0BFF),
    ("telugu",     0x0C00, 0x0C7F),
    ("kannada",    0x0C80, 0x0CFF),
    ("malayalam",  0x0D00, 0x0D7F),
    ("sinhala",    0x0D80, 0x0DFF),
];

pub fn detect_script_from_token_id(token_id: u32) -> Vec<f32> {
    let mut features = vec![0.0f32; 10];
    for (i, &(_, start, end)) in UNICODE_RANGES.iter().enumerate() {
        if token_id >= start && token_id <= end {
            if i < features.len() {
                features[i] = 1.0;
            }
            break;
        }
    }
    if token_id < 128 {
        features[9] = 1.0;
    }
    features
}

// ---------------------------------------------------------------------------
// Pre-trained model loader
// ---------------------------------------------------------------------------

pub fn load_pretrained_model(script: &str) -> Option<NeuralPredictor> {
    let vocab_size    = 5000;
    let embedding_dim = 32;
    let hidden_size   = 128;

    let mut model = NeuralPredictor::new(vocab_size, embedding_dim, hidden_size);

    // Zipf-law frequency tiers
    for i in 0..20.min(model.b2.len())   { model.b2[i]  =  3.5; }
    for i in 20..100.min(model.b2.len()) { model.b2[i]  =  2.0; }
    for i in 100..500.min(model.b2.len()){ model.b2[i]  =  0.5; }
    for i in 500..2000.min(model.b2.len()){ model.b2[i] = -2.0; }
    for i in 2000..4000.min(model.b2.len()){ model.b2[i]= -5.0; }
    for i in 4000..model.b2.len()         { model.b2[i] = -8.0; }

    match script.to_lowercase().as_str() {
        "devanagari" | "hindi" | "marathi" | "sanskrit" | "nepali" => tune_devanagari(&mut model),
        "tamil"                   => tune_tamil(&mut model),
        "telugu"                  => tune_telugu(&mut model),
        "kannada"                 => tune_kannada(&mut model),
        "malayalam"               => tune_malayalam(&mut model),
        "bengali" | "assamese" | "bangla" => tune_bengali(&mut model),
        "gujarati"                => tune_gujarati(&mut model),
        "punjabi" | "gurmukhi"    => tune_punjabi(&mut model),
        "odia" | "oriya"          => tune_odia(&mut model),
        "sinhala" | "sinhalese"   => tune_sinhala(&mut model),
        _                         => tune_universal_indic(&mut model),
    }

    // Pattern: local context dependency
    for h in 0..model.hidden_size {
        for o in 0..vocab_size.min(model.output_size) {
            let idx = h * model.output_size + o;
            if idx < model.w2.len() {
                let distance = ((h as i32) - (o as i32 / 40)).abs();
                let proximity = (-(distance as f32) / 30.0).exp();
                model.w2[idx] += proximity * 1.2;
            }
        }
    }

    // Pre-computed script embeddings
    for &(_, start, end) in UNICODE_RANGES {
        for token_id in start..=end.min(start + 200) {
            let script_features = detect_script_from_token_id(token_id);
            let mut embedding = vec![0.0f32; embedding_dim];
            for i in 0..embedding_dim {
                let pos_freq = ((token_id - start) as f32 / 50.0) * (i as f32 + 1.0);
                let pos_comp = (pos_freq.sin() * 0.35).tanh();
                let scr_comp = script_features[i % script_features.len()] * 0.25;
                let freq_comp = (-((token_id - start) as f32) / 100.0).exp() * 0.15;
                embedding[i] = pos_comp + scr_comp + freq_comp;
            }
            model.embeddings.insert(token_id, embedding);
        }
    }

    // ASCII / Latin embeddings
    for token_id in 32u32..=126 {
        let mut embedding = vec![0.0f32; embedding_dim];
        for i in 0..embedding_dim {
            let freq = (token_id as f32 / 12.0) * (i as f32 + 1.0);
            embedding[i] = (freq.cos() * 0.3).tanh();
        }
        model.embeddings.insert(token_id, embedding);
    }

    Some(model)
}

// ---------------------------------------------------------------------------
// Script-specific tuning helpers
// ---------------------------------------------------------------------------

fn tune_devanagari(model: &mut NeuralPredictor) {
    for code in 0x093Eu32..=0x094F { let off = (code - 0x0900) as usize; if off < model.b2.len() { model.b2[off] += 1.8; } }
    let ho = (0x094Du32 - 0x0900) as usize; if ho < model.b2.len() { model.b2[ho] += 2.2; }
    for code in 0x0915u32..=0x0939 { let off = (code - 0x0900) as usize; if off < model.b2.len() { model.b2[off] += 0.5; } }
}

fn tune_tamil(model: &mut NeuralPredictor) {
    let po = (0x0BCDu32 - 0x0B80) as usize + 200; if po < model.b2.len() { model.b2[po] += 1.5; }
    for code in 0x0BBEu32..=0x0BCC { let off = (code - 0x0B80) as usize + 200; if off < model.b2.len() { model.b2[off] += 1.2; } }
}

fn tune_telugu(model: &mut NeuralPredictor) {
    for code in 0x0C3Eu32..=0x0C4C { let off = (code - 0x0C00) as usize + 400; if off < model.b2.len() { model.b2[off] += 1.5; } }
    let vo = (0x0C4Du32 - 0x0C00) as usize + 400; if vo < model.b2.len() { model.b2[vo] += 1.8; }
}

fn tune_kannada(model: &mut NeuralPredictor) {
    for code in 0x0CBEu32..=0x0CCC { let off = (code - 0x0C80) as usize + 600; if off < model.b2.len() { model.b2[off] += 1.5; } }
    let vo = (0x0CCDu32 - 0x0C80) as usize + 600; if vo < model.b2.len() { model.b2[vo] += 1.8; }
}

fn tune_malayalam(model: &mut NeuralPredictor) {
    for code in 0x0D3Eu32..=0x0D4C { let off = (code - 0x0D00) as usize + 800; if off < model.b2.len() { model.b2[off] += 1.3; } }
    let vo = (0x0D4Du32 - 0x0D00) as usize + 800; if vo < model.b2.len() { model.b2[vo] += 2.0; }
}

fn tune_bengali(model: &mut NeuralPredictor) {
    for code in 0x09BEu32..=0x09CC { let off = (code - 0x0980) as usize + 1000; if off < model.b2.len() { model.b2[off] += 1.6; } }
    let ho = (0x09CDu32 - 0x0980) as usize + 1000; if ho < model.b2.len() { model.b2[ho] += 2.5; }
}

fn tune_gujarati(model: &mut NeuralPredictor) {
    for code in 0x0ABEu32..=0x0ACC { let off = (code - 0x0A80) as usize + 1200; if off < model.b2.len() { model.b2[off] += 1.4; } }
    let vo = (0x0ACDu32 - 0x0A80) as usize + 1200; if vo < model.b2.len() { model.b2[vo] += 1.9; }
}

fn tune_punjabi(model: &mut NeuralPredictor) {
    for code in 0x0A3Eu32..=0x0A4C { let off = (code - 0x0A00) as usize + 1400; if off < model.b2.len() { model.b2[off] += 1.3; } }
}

fn tune_odia(model: &mut NeuralPredictor) {
    for code in 0x0B3Eu32..=0x0B4C { let off = (code - 0x0B00) as usize + 1600; if off < model.b2.len() { model.b2[off] += 1.4; } }
    let vo = (0x0B4Du32 - 0x0B00) as usize + 1600; if vo < model.b2.len() { model.b2[vo] += 1.8; }
}

fn tune_sinhala(model: &mut NeuralPredictor) {
    for code in 0x0DCFu32..=0x0DDF { let off = (code - 0x0D80) as usize + 1800; if off < model.b2.len() { model.b2[off] += 1.3; } }
}

fn tune_universal_indic(model: &mut NeuralPredictor) {
    for i in 0..30.min(model.b2.len())  { model.b2[i] += 1.0; }
    for i in 40..60.min(model.b2.len()) { model.b2[i] += 0.6; }
}

