use std::collections::HashMap;

/// Multi-tier dictionary
/// - global list could be preloaded (not used in this minimal impl, but structure supports it)
pub struct MultiTierDict {
    pub global: Vec<String>,      // optional pre-seeded global tokens
    pub map: HashMap<String, u32>,// token -> id
    pub rev: HashMap<u32, String>,// id -> token
    pub next: u32,
}

impl MultiTierDict {
    pub fn new() -> Self {
        let global: Vec<String> = Vec::new();
        let mut map = HashMap::new();
        let mut rev = HashMap::new();
        // If you want seed global tokens: push here and set map/rev accordingly
        for (i, tok) in global.iter().enumerate() {
            map.insert(tok.clone(), i as u32);
            rev.insert(i as u32, tok.clone());
        }
        let next = global.len() as u32;
        Self { global, map, rev, next }
    }

    /// Get existing id or insert new id for token. Returns (id, was_new).
    pub fn get_or_insert(&mut self, token: &str) -> (u32, bool) {
        if let Some(&id) = self.map.get(token) {
            (id, false)
        } else {
            let id = self.next;
            self.map.insert(token.to_string(), id);
            self.rev.insert(id, token.to_string());
            self.next += 1;
            (id, true)
        }
    }

    /// Lookup token by id (may return None for unknown)
    pub fn lookup(&self, id: u32) -> Option<&String> {
        self.rev.get(&id)
    }

    /// Current symbol-space size (next id)
    pub fn next_id(&self) -> usize {
        self.next as usize
    }

    /// Add a delta entry (id -> token) — used by decoder when applying deltas
    pub fn add_with_id(&mut self, id: u32, token: &str) {
        // ensure `next` moves forward if id is larger
        self.map.insert(token.to_string(), id);
        self.rev.insert(id, token.to_string());
        if id >= self.next { self.next = id + 1; }
    }
}