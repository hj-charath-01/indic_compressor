// src/tokenize.rs
use unicode_segmentation::UnicodeSegmentation;

/// Tokenize into grapheme clusters using unicode-segmentation crate.
/// This is a practical and robust approach; we augment by grouping halant sequences if desired.
pub fn tokenize(text: &str) -> Vec<String> {
    // graphemes(true) yields extended grapheme clusters (close enough for Indic scripts)
    text.graphemes(true).map(|g| g.to_string()).collect()
}

/// Feature mask helper for a token (u8 bitmask)
/// Bits (suggested):
/// 0 - has_halant (virama)
/// 1 - has_combining_mark
/// 2 - starts_with_vowel (approx)
pub fn token_features(token: &str) -> u8 {
    let mut mask = 0u8;
    // Devanagari halant (virama) example (use brace form)
    if token.contains('\u{094D}') {
        mask |= 1;
    }
    // basic combining-mark heuristic: approximate common Indic combining ranges
    if token.chars().any(|c| {
        let cp = c as u32;
        // ranges include common combining marks across Indic blocks (heuristic)
        (cp >= 0x093C && cp <= 0x094D) || (cp >= 0x0ABC && cp <= 0x0B4D) || (cp >= 0x0C3C && cp <= 0x0C4D)
    }) {
        mask |= 2;
    }
    // vowel start heuristic (Devanagari vowel range example)
    if let Some(first) = token.chars().next() {
        let cp = first as u32;
        if cp >= 0x0904 && cp <= 0x0914 {
            mask |= 4;
        }
    }
    mask
}
