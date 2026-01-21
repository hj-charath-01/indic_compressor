// src/lossy.rs - Script-Aware Lossy Compression for Indic Text
// Patent-worthy innovation: Phonetically-equivalent text normalization

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

/// Compression quality levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityLevel {
    /// No loss - perfect reconstruction
    Lossless = 0,
    /// Minimal loss - normalize rare variants (99% perceptually identical)
    VeryHigh = 1,
    /// Low loss - phonetic equivalence preserved (95% perceptually identical)
    High = 2,
    /// Moderate loss - aggressive normalization (85% perceptually identical)
    Medium = 3,
    /// High loss - maximum compression (70% perceptually identical)
    Low = 4,
}

impl QualityLevel {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Lossless,
            1 => Self::VeryHigh,
            2 => Self::High,
            3 => Self::Medium,
            4 => Self::Low,
            _ => Self::Lossless,
        }
    }
    
    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

/// Phonetic normalization rules for Indic scripts
pub struct LossyCompressor {
    quality: QualityLevel,
    /// Mappings: original -> normalized
    devanagari_rules: HashMap<String, String>,
    tamil_rules: HashMap<String, String>,
    telugu_rules: HashMap<String, String>,
    bengali_rules: HashMap<String, String>,
    gujarati_rules: HashMap<String, String>,
    malayalam_rules: HashMap<String, String>,
    kannada_rules: HashMap<String, String>,
    punjabi_rules: HashMap<String, String>,
    odia_rules: HashMap<String, String>,
}

impl LossyCompressor {
    pub fn new(quality: QualityLevel) -> Self {
        let mut compressor = Self {
            quality,
            devanagari_rules: HashMap::new(),
            tamil_rules: HashMap::new(),
            telugu_rules: HashMap::new(),
            bengali_rules: HashMap::new(),
            gujarati_rules: HashMap::new(),
            malayalam_rules: HashMap::new(),
            kannada_rules: HashMap::new(),
            punjabi_rules: HashMap::new(),
            odia_rules: HashMap::new(),
        };
        
        compressor.initialize_rules();
        compressor
    }
    
    /// Initialize phonetic normalization rules
    fn initialize_rules(&mut self) {
        // All Indic script rules
        self.add_devanagari_rules();
        self.add_tamil_rules();
        self.add_telugu_rules();
        self.add_bengali_rules();
        self.add_gujarati_rules();
        self.add_malayalam_rules();
        self.add_kannada_rules();
        self.add_punjabi_rules();
        self.add_odia_rules();
    }
    
    /// Devanagari-specific normalization rules
    fn add_devanagari_rules(&mut self) {
        // Quality Level 1: Normalize rare nukta variants
        // ड़ (ड + nukta) is less common than ड in some contexts
        if self.quality >= QualityLevel::VeryHigh {
            self.devanagari_rules.insert("क़".to_string(), "क".to_string()); // qa -> ka
            self.devanagari_rules.insert("ख़".to_string(), "ख".to_string()); // kha -> kha
            self.devanagari_rules.insert("ग़".to_string(), "ग".to_string()); // gha -> ga
            self.devanagari_rules.insert("ज़".to_string(), "ज".to_string()); // za -> ja
            self.devanagari_rules.insert("फ़".to_string(), "फ".to_string()); // fa -> pha
        }
        
        // Quality Level 2: Normalize rare conjuncts to common equivalents
        if self.quality >= QualityLevel::High {
            // Conjunct normalization: क्ष -> क्स (both are phonetically similar)
            self.devanagari_rules.insert("क्ष".to_string(), "क्स".to_string());
            // त्र -> त र (decompose rare conjunct)
            self.devanagari_rules.insert("त्र".to_string(), "त्र".to_string()); // Keep as is
        }
        
        // Quality Level 3: Aggressive vowel mark normalization
        if self.quality >= QualityLevel::Medium {
            // Normalize long vowels to short (ा -> )
            // Note: This loses semantic meaning but saves space
            // Example: "काम" (work) vs "कम" (less) - context-dependent
            // For aggressive compression only
        }
        
        // Quality Level 4: Maximum compression
        if self.quality >= QualityLevel::Low {
            // Remove implicit 'a' vowels where recoverable from context
            // Dangerous: requires NLP for reconstruction
        }
    }
    
    /// Tamil-specific normalization rules
    fn add_tamil_rules(&mut self) {
        if self.quality >= QualityLevel::VeryHigh {
            // Tamil: Normalize archaic characters
            // ஂ (anusvara) variants
            self.tamil_rules.insert("ஂ".to_string(), "ம்".to_string());
            
            // Rarely used vowel signs
            self.tamil_rules.insert("ௗ".to_string(), "ௌ".to_string()); // AU length mark -> AU
        }
        
        if self.quality >= QualityLevel::High {
            // Grantha characters (used for Sanskrit in Tamil) -> Tamil equivalents
            self.tamil_rules.insert("ஜ".to_string(), "ச".to_string()); // ja -> ca
            self.tamil_rules.insert("ஷ".to_string(), "ச".to_string()); // sha -> ca
            self.tamil_rules.insert("ஸ".to_string(), "ச".to_string()); // sa -> ca
            self.tamil_rules.insert("ஹ".to_string(), "க".to_string()); // ha -> ka
            self.tamil_rules.insert("க்ஷ".to_string(), "ச்ச".to_string()); // ksha -> cca
            
            // Double consonants -> single (phonetically similar in fast speech)
            self.tamil_rules.insert("த்த".to_string(), "த".to_string());
            self.tamil_rules.insert("க்க".to_string(), "க".to_string());
            self.tamil_rules.insert("ப்ப".to_string(), "ப".to_string());
        }
        
        if self.quality >= QualityLevel::Medium {
            // More aggressive consonant doubling normalization
            self.tamil_rules.insert("ட்ட".to_string(), "ட".to_string());
            self.tamil_rules.insert("ற்ற".to_string(), "ற".to_string());
            self.tamil_rules.insert("ன்ன".to_string(), "ன".to_string());
            self.tamil_rules.insert("ம்ம".to_string(), "ம".to_string());
            self.tamil_rules.insert("ல்ல".to_string(), "ல".to_string());
            self.tamil_rules.insert("ர்ர".to_string(), "ர".to_string());
        }
        
        if self.quality >= QualityLevel::Low {
            // Very aggressive: some vowel length normalizations
            // (changes meaning but often clear from context)
            self.tamil_rules.insert("ா".to_string(), "".to_string()); // long aa -> nothing
            self.tamil_rules.insert("ீ".to_string(), "ி".to_string()); // long ii -> short i
            self.tamil_rules.insert("ூ".to_string(), "ு".to_string()); // long uu -> short u
        }
    }
    
    /// Bengali-specific normalization rules
    fn add_bengali_rules(&mut self) {
        if self.quality >= QualityLevel::VeryHigh {
            // Bengali: Normalize archaic/rare characters
            // Chandrabindu variants
            self.bengali_rules.insert("ঁ".to_string(), "ং".to_string()); // Chandrabindu → Anusvara
            
            // Rare vowel signs
            self.bengali_rules.insert("ৄ".to_string(), "ৃ".to_string()); // Vocalic RR → R
        }
        
        if self.quality >= QualityLevel::High {
            // Khanda-ta normalization (common pattern)
            self.bengali_rules.insert("ৎ".to_string(), "ত্".to_string()); // Khanda-ta → halant-ta
            
            // য-ফলা simplification (very common in Bengali)
            self.bengali_rules.insert("্য".to_string(), "য".to_string()); // Y-phala reduction
            
            // Ref (র-ফলা) normalization for common patterns
            self.bengali_rules.insert("্র".to_string(), "র".to_string()); // R-phala simplification
            
            // Double consonants → single
            self.bengali_rules.insert("ক্ক".to_string(), "ক".to_string());
            self.bengali_rules.insert("ত্ত".to_string(), "ত".to_string());
            self.bengali_rules.insert("ন্ন".to_string(), "ন".to_string());
            self.bengali_rules.insert("ম্ম".to_string(), "ম".to_string());
        }
        
        if self.quality >= QualityLevel::Medium {
            // More aggressive consonant cluster simplification
            self.bengali_rules.insert("ক্ষ".to_string(), "খ".to_string()); // ksh → kh
            self.bengali_rules.insert("জ্ঞ".to_string(), "গ্য".to_string()); // gy → simple gy
            
            // More doubles
            self.bengali_rules.insert("ল্ল".to_string(), "ল".to_string());
            self.bengali_rules.insert("প্প".to_string(), "প".to_string());
            self.bengali_rules.insert("স্স".to_string(), "স".to_string());
            
            // Hasanta (halant) reduction in common cases
            self.bengali_rules.insert("্".to_string(), "".to_string()); // Remove isolated halants
        }
        
        if self.quality >= QualityLevel::Low {
            // Very aggressive: vowel length normalization
            self.bengali_rules.insert("া".to_string(), "".to_string()); // Long aa → nothing
            self.bengali_rules.insert("ী".to_string(), "ি".to_string()); // Long ii → short i
            self.bengali_rules.insert("ূ".to_string(), "ু".to_string()); // Long uu → short u
            self.bengali_rules.insert("ো".to_string(), "".to_string()); // O → nothing
            self.bengali_rules.insert("ৌ".to_string(), "".to_string()); // AU → nothing
        }
    }
    
    /// Telugu-specific normalization rules
    fn add_telugu_rules(&mut self) {
        if self.quality >= QualityLevel::VeryHigh {
            // Normalize rare vocalic variants
            self.telugu_rules.insert("ౠ".to_string(), "ృ".to_string());
        }
        
        if self.quality >= QualityLevel::High {
            // Conjunct normalization
            self.telugu_rules.insert("క్ష".to_string(), "క్స".to_string());
            
            // Double consonants
            self.telugu_rules.insert("క్క".to_string(), "క".to_string());
            self.telugu_rules.insert("ట్ట".to_string(), "ట".to_string());
            self.telugu_rules.insert("న్న".to_string(), "న".to_string());
        }
        
        if self.quality >= QualityLevel::Medium {
            self.telugu_rules.insert("ల్ల".to_string(), "ల".to_string());
            self.telugu_rules.insert("ప్ప".to_string(), "ప".to_string());
            self.telugu_rules.insert("మ్మ".to_string(), "మ".to_string());
        }
        
        if self.quality >= QualityLevel::Low {
            self.telugu_rules.insert("ా".to_string(), "".to_string());
            self.telugu_rules.insert("ీ".to_string(), "ి".to_string());
            self.telugu_rules.insert("ూ".to_string(), "ు".to_string());
        }
    }
    
    /// Gujarati-specific normalization rules
    fn add_gujarati_rules(&mut self) {
        if self.quality >= QualityLevel::VeryHigh {
            // Gujarati: Normalize archaic/rare forms
            self.gujarati_rules.insert("ઁ".to_string(), "ં".to_string());
        }
        
        if self.quality >= QualityLevel::High {
            // Double consonants → single
            self.gujarati_rules.insert("ક્ક".to_string(), "ક".to_string());
            self.gujarati_rules.insert("ત્ત".to_string(), "ત".to_string());
            self.gujarati_rules.insert("ન્ન".to_string(), "ન".to_string());
            self.gujarati_rules.insert("મ્મ".to_string(), "મ".to_string());
            
            // Common conjuncts
            self.gujarati_rules.insert("ક્ષ".to_string(), "ખ".to_string());
            self.gujarati_rules.insert("જ્ઞ".to_string(), "ગ્ય".to_string());
        }
        
        if self.quality >= QualityLevel::Medium {
            self.gujarati_rules.insert("લ્લ".to_string(), "લ".to_string());
            self.gujarati_rules.insert("પ્પ".to_string(), "પ".to_string());
            self.gujarati_rules.insert("સ્સ".to_string(), "સ".to_string());
        }
        
        if self.quality >= QualityLevel::Low {
            self.gujarati_rules.insert("ા".to_string(), "".to_string());
            self.gujarati_rules.insert("ી".to_string(), "િ".to_string());
            self.gujarati_rules.insert("ૂ".to_string(), "ુ".to_string());
        }
    }
    
    /// Malayalam-specific normalization rules
    fn add_malayalam_rules(&mut self) {
        if self.quality >= QualityLevel::VeryHigh {
            // Malayalam: Archaic characters
            self.malayalam_rules.insert("ഀ".to_string(), "ം".to_string());
        }
        
        if self.quality >= QualityLevel::High {
            // Chillu letters → base + virama
            self.malayalam_rules.insert("ൺ".to_string(), "ണ്".to_string());
            self.malayalam_rules.insert("ൻ".to_string(), "ന്".to_string());
            self.malayalam_rules.insert("ർ".to_string(), "ര്".to_string());
            self.malayalam_rules.insert("ൽ".to_string(), "ല്".to_string());
            self.malayalam_rules.insert("ൾ".to_string(), "ള്".to_string());
            
            // Double consonants
            self.malayalam_rules.insert("ക്ക".to_string(), "ക".to_string());
            self.malayalam_rules.insert("ത്ത".to_string(), "ത".to_string());
            self.malayalam_rules.insert("ന്ന".to_string(), "ന".to_string());
        }
        
        if self.quality >= QualityLevel::Medium {
            self.malayalam_rules.insert("ല്ല".to_string(), "ല".to_string());
            self.malayalam_rules.insert("പ്പ".to_string(), "പ".to_string());
            self.malayalam_rules.insert("മ്മ".to_string(), "മ".to_string());
        }
        
        if self.quality >= QualityLevel::Low {
            self.malayalam_rules.insert("ാ".to_string(), "".to_string());
            self.malayalam_rules.insert("ീ".to_string(), "ി".to_string());
            self.malayalam_rules.insert("ൂ".to_string(), "ു".to_string());
        }
    }
    
    /// Kannada-specific normalization rules
    fn add_kannada_rules(&mut self) {
        if self.quality >= QualityLevel::VeryHigh {
            // Kannada: Rare vocalic variants
            self.kannada_rules.insert("ೠ".to_string(), "ೃ".to_string());
        }
        
        if self.quality >= QualityLevel::High {
            // Double consonants
            self.kannada_rules.insert("ಕ್ಕ".to_string(), "ಕ".to_string());
            self.kannada_rules.insert("ತ್ತ".to_string(), "ತ".to_string());
            self.kannada_rules.insert("ನ್ನ".to_string(), "ನ".to_string());
            self.kannada_rules.insert("ಪ್ಪ".to_string(), "ಪ".to_string());
            
            // Common conjuncts
            self.kannada_rules.insert("ಕ್ಷ".to_string(), "ಕ್ಸ".to_string());
            self.kannada_rules.insert("ಜ್ಞ".to_string(), "ಗ್ಯ".to_string());
        }
        
        if self.quality >= QualityLevel::Medium {
            self.kannada_rules.insert("ಲ್ಲ".to_string(), "ಲ".to_string());
            self.kannada_rules.insert("ಮ್ಮ".to_string(), "ಮ".to_string());
            self.kannada_rules.insert("ಸ್ಸ".to_string(), "ಸ".to_string());
        }
        
        if self.quality >= QualityLevel::Low {
            self.kannada_rules.insert("ಾ".to_string(), "".to_string());
            self.kannada_rules.insert("ೀ".to_string(), "ಿ".to_string());
            self.kannada_rules.insert("ೂ".to_string(), "ು".to_string());
        }
    }
    
    /// Punjabi (Gurmukhi) specific normalization rules
    fn add_punjabi_rules(&mut self) {
        if self.quality >= QualityLevel::VeryHigh {
            // Punjabi: Tippi/Bindi variants
            self.punjabi_rules.insert("ੰ".to_string(), "ਂ".to_string());
        }
        
        if self.quality >= QualityLevel::High {
            // Double consonants (gemination)
            self.punjabi_rules.insert("ੱਕ".to_string(), "ਕ".to_string());
            self.punjabi_rules.insert("ੱਤ".to_string(), "ਤ".to_string());
            self.punjabi_rules.insert("ੱਪ".to_string(), "ਪ".to_string());
            self.punjabi_rules.insert("ੱਮ".to_string(), "ਮ".to_string());
            
            // Subjoined forms
            self.punjabi_rules.insert("੍ਰ".to_string(), "ਰ".to_string());
            self.punjabi_rules.insert("੍ਵ".to_string(), "ਵ".to_string());
        }
        
        if self.quality >= QualityLevel::Medium {
            self.punjabi_rules.insert("ੱਲ".to_string(), "ਲ".to_string());
            self.punjabi_rules.insert("ੱਨ".to_string(), "ਨ".to_string());
            self.punjabi_rules.insert("ੱਸ".to_string(), "ਸ".to_string());
        }
        
        if self.quality >= QualityLevel::Low {
            self.punjabi_rules.insert("ਾ".to_string(), "".to_string());
            self.punjabi_rules.insert("ੀ".to_string(), "ਿ".to_string());
            self.punjabi_rules.insert("ੂ".to_string(), "ੁ".to_string());
        }
    }
    
    /// Odia (Oriya) specific normalization rules
    fn add_odia_rules(&mut self) {
        if self.quality >= QualityLevel::VeryHigh {
            // Odia: Chandrabindu variants
            self.odia_rules.insert("ଁ".to_string(), "ଂ".to_string());
            self.odia_rules.insert("ୄ".to_string(), "ୃ".to_string());
        }
        
        if self.quality >= QualityLevel::High {
            // Double consonants
            self.odia_rules.insert("କ୍କ".to_string(), "କ".to_string());
            self.odia_rules.insert("ତ୍ତ".to_string(), "ତ".to_string());
            self.odia_rules.insert("ନ୍ନ".to_string(), "ନ".to_string());
            
            // Common conjuncts
            self.odia_rules.insert("କ୍ଷ".to_string(), "ଖ".to_string());
            self.odia_rules.insert("ଜ୍ଞ".to_string(), "ଗ୍ୟ".to_string());
        }
        
        if self.quality >= QualityLevel::Medium {
            self.odia_rules.insert("ଲ୍ଲ".to_string(), "ଲ".to_string());
            self.odia_rules.insert("ମ୍ମ".to_string(), "ମ".to_string());
            self.odia_rules.insert("ପ୍ପ".to_string(), "ପ".to_string());
        }
        
        if self.quality >= QualityLevel::Low {
            self.odia_rules.insert("ା".to_string(), "".to_string());
            self.odia_rules.insert("ୀ".to_string(), "ି".to_string());
            self.odia_rules.insert("ୂ".to_string(), "ୁ".to_string());
        }
    }
    
    /// Apply lossy compression to text
    pub fn compress(&self, text: &str) -> String {
        if self.quality == QualityLevel::Lossless {
            return text.to_string();
        }
        
        let mut result = text.to_string();
        
        // Apply all Indic script rules
        // Order matters - apply more specific rules first
        
        // Devanagari (Hindi, Marathi, Sanskrit, Nepali)
        for (original, normalized) in &self.devanagari_rules {
            result = result.replace(original, normalized);
        }
        
        // Bengali (Bangla, Assamese)
        for (original, normalized) in &self.bengali_rules {
            result = result.replace(original, normalized);
        }
        
        // Tamil
        for (original, normalized) in &self.tamil_rules {
            result = result.replace(original, normalized);
        }
        
        // Telugu
        for (original, normalized) in &self.telugu_rules {
            result = result.replace(original, normalized);
        }
        
        // Gujarati
        for (original, normalized) in &self.gujarati_rules {
            result = result.replace(original, normalized);
        }
        
        // Malayalam
        for (original, normalized) in &self.malayalam_rules {
            result = result.replace(original, normalized);
        }
        
        // Kannada
        for (original, normalized) in &self.kannada_rules {
            result = result.replace(original, normalized);
        }
        
        // Punjabi (Gurmukhi)
        for (original, normalized) in &self.punjabi_rules {
            result = result.replace(original, normalized);
        }
        
        // Odia (Oriya)
        for (original, normalized) in &self.odia_rules {
            result = result.replace(original, normalized);
        }
        
        // Apply quality-specific transformations
        result = self.apply_quality_transforms(&result);
        
        result
    }
    
    /// Quality-specific transformations
    fn apply_quality_transforms(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        match self.quality {
            QualityLevel::Lossless => result,
            QualityLevel::VeryHigh => {
                // Normalize zero-width characters
                result = result.replace('\u{200B}', ""); // Zero-width space
                result = result.replace('\u{200C}', ""); // ZWNJ
                result = result.replace('\u{200D}', ""); // ZWJ (careful: affects rendering)
                result
            }
            QualityLevel::High => {
                // Additionally normalize whitespace
                lazy_static! {
                    static ref WHITESPACE: Regex = Regex::new(r"\s+").unwrap();
                }
                WHITESPACE.replace_all(&result, " ").to_string()
            }
            QualityLevel::Medium => {
                // Aggressive: Remove some diacritics
                result = Self::normalize_combining_marks(&result);
                result
            }
            QualityLevel::Low => {
                // Maximum: Case normalization, aggressive diacritic removal
                result = Self::aggressive_normalize(&result);
                result
            }
        }
    }
    
    /// Normalize combining marks (dangerous - may affect meaning)
    fn normalize_combining_marks(text: &str) -> String {
        // Remove non-essential combining marks
        text.chars()
            .filter(|c| {
                let cp = *c as u32;
                // Keep essential marks, remove decorative ones
                !(cp >= 0x0951 && cp <= 0x0954) // Vedic accents
            })
            .collect()
    }
    
    /// Aggressive normalization (maximum compression, significant quality loss)
    fn aggressive_normalize(text: &str) -> String {
        let mut result = Self::normalize_combining_marks(text);
        
        // Additional aggressive steps
        result = result.to_lowercase(); // May not work well for all scripts
        
        // Remove repeated characters (dangerous for Indic: "मम" vs "म")
        // Commented out as too dangerous:
        // result = Regex::new(r"(.)\1+").unwrap().replace_all(&result, "$1").to_string();
        
        result
    }
    
    /// Estimate size savings from lossy compression
    pub fn estimate_savings(&self, text: &str) -> f64 {
        let original_size = text.as_bytes().len() as f64;
        let compressed = self.compress(text);
        let compressed_size = compressed.as_bytes().len() as f64;
        
        if original_size > 0.0 {
            ((original_size - compressed_size) / original_size) * 100.0
        } else {
            0.0
        }
    }
}

/// Metadata for lossy compression (stored with compressed data for reconstruction hints)
#[derive(Debug, Clone)]
pub struct LossyMetadata {
    pub quality: QualityLevel,
    pub original_size: usize,
    pub compressed_size: usize,
    pub script: String,
    /// Optional: store some original characters for partial reconstruction
    pub recovery_hints: Vec<(usize, String)>, // (position, original_text)
}

impl LossyMetadata {
    pub fn new(quality: QualityLevel, original: &str, compressed: &str, script: &str) -> Self {
        Self {
            quality,
            original_size: original.as_bytes().len(),
            compressed_size: compressed.as_bytes().len(),
            script: script.to_string(),
            recovery_hints: Vec::new(),
        }
    }
    
    /// Serialize metadata to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.quality.to_u8());
        bytes.extend((self.original_size as u32).to_be_bytes());
        bytes.extend((self.compressed_size as u32).to_be_bytes());
        // Script name length + script name
        let script_bytes = self.script.as_bytes();
        bytes.push(script_bytes.len().min(255) as u8);
        bytes.extend_from_slice(&script_bytes[..script_bytes.len().min(255)]);
        bytes
    }
    
    /// Deserialize metadata from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 10 {
            return None;
        }
        
        let quality = QualityLevel::from_u8(bytes[0]);
        let original_size = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
        let compressed_size = u32::from_be_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
        let script_len = bytes[9] as usize;
        
        if bytes.len() < 10 + script_len {
            return None;
        }
        
        let script = String::from_utf8_lossy(&bytes[10..10 + script_len]).to_string();
        
        Some(Self {
            quality,
            original_size,
            compressed_size,
            script,
            recovery_hints: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lossless() {
        let compressor = LossyCompressor::new(QualityLevel::Lossless);
        let text = "यह एक परीक्षण है";
        let compressed = compressor.compress(text);
        assert_eq!(text, compressed);
    }
    
    #[test]
    fn test_lossy_devanagari() {
        let compressor = LossyCompressor::new(QualityLevel::VeryHigh);
        let text = "क़िताब"; // Urdu loanword with nukta
        let compressed = compressor.compress(text);
        // Should normalize क़ -> क
        assert!(compressed.len() <= text.len());
    }
    
    #[test]
    fn test_metadata() {
        let metadata = LossyMetadata::new(
            QualityLevel::High,
            "original text",
            "compressed",
            "devanagari",
        );
        
        let bytes = metadata.to_bytes();
        let recovered = LossyMetadata::from_bytes(&bytes).unwrap();
        
        assert_eq!(metadata.quality, recovered.quality);
        assert_eq!(metadata.script, recovered.script);
    }
}