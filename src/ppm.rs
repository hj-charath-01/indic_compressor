use std::collections::HashMap;

/// Simple PPM-like model (order-2) with frequency tables per context.
/// For prototype clarity we use a ContextKey as Vec<u32> with features appended.
pub type Symbol = u32;

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct ContextKey {
    pub keys: Vec<u32>, // prev symbols (length up to order) followed by features as u32
}

impl ContextKey {
    pub fn new(prev: &[u32], feature_mask: u8) -> Self {
        let mut keys = Vec::with_capacity(prev.len()+1);
        for &p in prev {
            keys.push(p);
        }
        keys.push(feature_mask as u32);
        ContextKey { keys }
    }
}

pub struct PPMModel {
    // For each context we hold a vector of counts indexed by symbol id (size = max_symbols)
    table: HashMap<ContextKey, Vec<u32>>,
    global: Vec<u32>,
    max_symbols: usize,
}

impl PPMModel {
    pub fn new(_order: usize, max_symbols: usize) -> Self {
        PPMModel {
            table: HashMap::new(),
            global: vec![1u32; max_symbols], // Laplace smoothing - start with 1
            max_symbols,
        }
    }

    /// Get probability (frequency) vector for a context. Returns Vec<u32> (copied) and total.
    pub fn get_freqs(&self, prev: &[u32], feature_mask: u8) -> (Vec<u32>, u32) {
        // try longest context then fallback to global
        let ctx = ContextKey::new(prev, feature_mask);
        if let Some(v) = self.table.get(&ctx) {
            let out = v.clone();
            let total: u32 = out.iter().sum();
            return (out, total);
        }
        // fallback: return global
        let out = self.global.clone();
        let total: u32 = out.iter().sum();
        (out, total)
    }

    /// Update model counts for a (prev, feature, symbol)
    pub fn update(&mut self, prev: &[u32], feature_mask: u8, symbol: Symbol) {
        // update context table
        let ctx = ContextKey::new(prev, feature_mask);
        let entry = self.table.entry(ctx).or_insert_with(|| vec![1u32; self.max_symbols]);
        let idx = symbol as usize;
        if idx >= entry.len() {
            // resize (shouldn't normally happen if max_symbols correct)
            entry.resize(self.max_symbols, 1u32);
        }
        entry[idx] += 1;
        // update global
        if symbol as usize >= self.global.len() {
            self.global.resize(self.max_symbols, 1u32);
        }
        self.global[symbol as usize] += 1;
        // Note: memory can grow; for production implement LRU/limit on table size.
    }
}