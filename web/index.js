// Enhanced Indic Compressor - Neural & Lossy Compression Interface
import init, { 
  encode_stream_wasm,
  encode_stream_advanced_wasm, 
  decode_prefix_wasm, 
  decode_full_wasm,
  estimate_lossy_savings 
} from './pkg/indic_ans_compressor.js';

// State
let wasmReady = false;
let latestStreamBytes = null;
let latestOriginalText = '';
let currentQuality = 0; // Lossless by default

// Safe DOM operations
function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = String(text);
    return true;
  }
  return false;
}

function safeSetHTML(id, html) {
  const el = document.getElementById(id);
  if (el) {
    el.innerHTML = String(html);
    return true;
  }
  return false;
}

function safeSetDisplay(id, display) {
  const el = document.getElementById(id);
  if (el && el.style) {
    el.style.display = display;
    return true;
  }
  return false;
}

function safeGetValue(id) {
  const el = document.getElementById(id);
  return (el && 'value' in el) ? el.value : '';
}

function showToast(message, type = 'info') {
  const toast = document.getElementById('statusToast');
  const toastMsg = document.getElementById('toastMessage');
  
  if (toastMsg) toastMsg.textContent = message;
  if (toast) {
    toast.className = `toast ${type}`;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 3000);
  }
}

function updateStats() {
  try {
    const text = safeGetValue('inputText');
    const chars = text.length;
    const bytes = new TextEncoder().encode(text).length;
    safeSetText('charCount', chars.toLocaleString());
    safeSetText('byteCount', bytes.toLocaleString());
    
    // Update lossy savings estimate
    updateLossySavings();
  } catch (e) {
    console.error('Error updating stats:', e);
  }
}

async function updateLossySavings() {
  if (!wasmReady) return;
  
  const text = safeGetValue('inputText');
  if (!text) {
    safeSetHTML('lossySavings', 'Estimated lossy savings: <strong>0.0%</strong>');
    return;
  }
  
  try {
    const savings = await estimate_lossy_savings(text, currentQuality);
    safeSetHTML('lossySavings', 
      `Estimated lossy savings: <strong>${savings}</strong>`);
  } catch (e) {
    console.error('Error estimating lossy savings:', e);
  }
}

function handleClear() {
  const input = document.getElementById('inputText');
  if (input) input.value = '';
  updateStats();
  hideResults();
  showToast('Cleared', 'success');
}

function hideResults() {
  safeSetDisplay('resultsPanel', 'none');
  safeSetDisplay('decodePanel', 'none');
  safeSetDisplay('outputPanel', 'none');
  latestStreamBytes = null;
}

// Initialize controls
(function initControls() {
  // Chunk size slider
  const chunkSize = document.getElementById('chunkSize');
  const chunkSizeDisplay = document.getElementById('chunkSizeDisplay');
  if (chunkSize && chunkSizeDisplay) {
    chunkSizeDisplay.textContent = chunkSize.value;
    chunkSize.addEventListener('input', () => {
      chunkSizeDisplay.textContent = chunkSize.value;
      updateSliderFill(chunkSize);
    });
    updateSliderFill(chunkSize);
  }

  // Neural weight slider
  const neuralWeight = document.getElementById('neuralWeight');
  const neuralWeightDisplay = document.getElementById('neuralWeightDisplay');
  if (neuralWeight && neuralWeightDisplay) {
    neuralWeightDisplay.textContent = neuralWeight.value + '%';
    neuralWeight.addEventListener('input', () => {
      neuralWeightDisplay.textContent = neuralWeight.value + '%';
      updateSliderFill(neuralWeight);
    });
    updateSliderFill(neuralWeight);
  }

  // Neural toggle
  const useNeural = document.getElementById('useNeural');
  const neuralStatus = document.getElementById('neuralStatus');
  const neuralWeightGroup = document.getElementById('neuralWeightGroup');
  if (useNeural && neuralStatus && neuralWeightGroup) {
    useNeural.addEventListener('change', () => {
      if (useNeural.checked) {
        neuralStatus.textContent = 'Enabled 🧠';
        neuralStatus.style.color = '#667eea';
        neuralStatus.style.fontWeight = '700';
        neuralWeightGroup.style.display = 'block';
      } else {
        neuralStatus.textContent = 'Disabled';
        neuralStatus.style.color = 'var(--text-muted)';
        neuralStatus.style.fontWeight = '400';
        neuralWeightGroup.style.display = 'none';
      }
    });
  }

  // Quality selector
  const qualityOptions = document.querySelectorAll('.quality-option');
  qualityOptions.forEach(option => {
    option.addEventListener('click', () => {
      qualityOptions.forEach(opt => opt.classList.remove('active'));
      option.classList.add('active');
      currentQuality = parseInt(option.getAttribute('data-quality'));
      updateLossySavings();
    });
  });

  // Decode slider
  const chunkToDecode = document.getElementById('chunkToDecode');
  const chunkToDecodeDisplay = document.getElementById('chunkToDecodeDisplay');
  if (chunkToDecode && chunkToDecodeDisplay) {
    chunkToDecodeDisplay.textContent = chunkToDecode.value;
    chunkToDecode.addEventListener('input', () => {
      chunkToDecodeDisplay.textContent = chunkToDecode.value;
      updateSliderFill(chunkToDecode);
    });
    updateSliderFill(chunkToDecode);
  }
})();

function updateSliderFill(slider) {
  const min = Number(slider.min || 0);
  const max = Number(slider.max || 100);
  const val = Number(slider.value);
  const percent = max === min ? 0 : (val - min) / (max - min) * 100;
  slider.style.background = `linear-gradient(90deg, var(--primary) ${percent}%, var(--border) ${percent}%)`;
}

function countChunksFromBinary(u8) {
  if (!u8 || u8.length < 2) return 0;
  
  // Check format
  if (u8[0] === 0x55 && u8[1] === 0x43) return 1; // "UC"
  if (u8[0] === 0x55 && u8[1] === 0x4C) return 1; // "UL"
  
  // Skip metadata if present
  let offset = 0;
  if (u8[0] === 0x49 && u8[1] === 0x4C) { // "IL"
    if (u8.length < 3) return 0;
    const metaLen = u8[2];
    offset = 3 + metaLen;
  }
  
  // Count IC chunks
  if (u8[offset] !== 0x49 || u8[offset + 1] !== 0x43) return 0;

  const dv = new DataView(u8.buffer, u8.byteOffset + offset, u8.byteLength - offset);
  let pos = 0;
  let count = 0;
  
  try {
    while (pos + 2 <= u8.length - offset) {
      if (u8[offset + pos] !== 0x49 || u8[offset + pos + 1] !== 0x43) break;
      pos += 2;
      
      if (pos + 2 > u8.length - offset) break;
      const token_count = dv.getUint16(pos, false);
      pos += 2;
      
      if (pos + 1 > u8.length - offset) break;
      const delta_count = u8[offset + pos];
      pos += 1;
      
      for (let i = 0; i < delta_count; i++) {
        if (pos + 3 > u8.length - offset) { pos = u8.length - offset; break; }
        pos += 2;
        const len = u8[offset + pos];
        pos += 1;
        if (pos + len > u8.length - offset) { pos = u8.length - offset; break; }
        pos += len;
      }
      
      if (pos + token_count > u8.length - offset) break;
      pos += token_count;
      
      if (pos + 2 > u8.length - offset) break;
      const payload_len = dv.getUint16(pos, false);
      pos += 2;
      
      if (pos + payload_len > u8.length - offset) break;
      pos += payload_len;
      
      count++;
    }
  } catch (e) {
    console.error('Error parsing chunks:', e);
  }
  
  return count;
}

function getCompressionFormat(u8) {
  if (!u8 || u8.length < 2) return 'unknown';
  const magic = String.fromCharCode(u8[0], u8[1]);
  switch (magic) {
    case 'UC': return 'uncompressed';
    case 'UL': return 'uncompressed-lossy';
    case 'IC': return 'compressed';
    case 'IL': return 'compressed-lossy';
    default: return 'unknown';
  }
}

async function handleEncode() {
  try {
    if (!wasmReady) {
      showToast('⏳ Please wait, loading...', 'warning');
      return;
    }
    
    const text = safeGetValue('inputText').trim();
    if (!text) {
      showToast('Please enter some text to compress', 'warning');
      return;
    }
    
    latestOriginalText = text;
    const originalBytes = new TextEncoder().encode(text).length;
    const chunkSize = Number(safeGetValue('chunkSize')) || 40;
    const useNeural = document.getElementById('useNeural')?.checked || false;
    const neuralWeight = Number(safeGetValue('neuralWeight')) / 100.0;
    
    const btn = document.getElementById('btnEncode');
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Compressing...';
    }
    
    let encoded;
    if (useNeural || currentQuality > 0) {
      // Use advanced compression with neural/lossy features
      encoded = encode_stream_advanced_wasm(
        text,
        chunkSize,
        useNeural,
        neuralWeight,
        currentQuality
      );
    } else {
      // Use standard compression
      encoded = encode_stream_wasm(text, chunkSize);
    }
    
    const u8 = encoded instanceof Uint8Array ? encoded : new Uint8Array(encoded);
    latestStreamBytes = u8;
    
    const compressedBytes = u8.length;
    const ratio = (compressedBytes / Math.max(1, originalBytes) * 100).toFixed(1);
    const chunks = countChunksFromBinary(u8);
    const format = getCompressionFormat(u8);
    
    const savings = originalBytes - compressedBytes;
    
    safeSetText('originalSize', `${originalBytes.toLocaleString()} bytes`);
    safeSetText('compressedSize', `${compressedBytes.toLocaleString()} bytes`);
    
    if (savings > 0) {
      safeSetText('compressionRatio', `${ratio}% (saved ${savings.toLocaleString()} bytes)`);
      showToast(`✓ Compressed to ${ratio}% of original size`, 'success');
    } else {
      safeSetText('compressionRatio', `${ratio}%`);
      showToast(`⚠ Result is ${ratio}% of original`, 'warning');
    }
    
    // Show method used
    let method = format;
    if (useNeural) method += ' + neural';
    if (currentQuality > 0) method += ' + lossy';
    safeSetText('methodUsed', method);
    
    safeSetDisplay('resultsPanel', 'block');
    if (chunks > 1) {
      const slider = document.getElementById('chunkToDecode');
      if (slider) {
        slider.max = chunks;
        slider.value = chunks;
        safeSetText('chunkToDecodeDisplay', chunks);
        updateSliderFill(slider);
      }
      safeSetDisplay('decodePanel', 'block');
    }
    
    const downloadBtn = document.getElementById('btnDownload');
    if (downloadBtn) {
      downloadBtn.onclick = () => {
        const blob = new Blob([u8], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'indic_compressed_enhanced.bin';
        a.click();
        URL.revokeObjectURL(url);
        showToast('✓ File downloaded', 'success');
      };
    }
    
  } catch (e) {
    console.error('Encode error:', e);
    showToast(`❌ Compression failed: ${e.message || e}`, 'error');
  } finally {
    const btn = document.getElementById('btnEncode');
    if (btn) {
      btn.disabled = false;
      btn.textContent = '🗜️ Compress Text';
    }
  }
}

async function handleDecodeAll() {
  try {
    if (!latestStreamBytes) {
      showToast('No compressed data available', 'warning');
      return;
    }
    
    const btn = document.getElementById('btnDecodeAll');
    if (btn) { btn.disabled = true; btn.textContent = 'Decompressing...'; }
    
    const decoded = decode_full_wasm(latestStreamBytes);
    
    safeSetText('decodedOutput', decoded);
    safeSetDisplay('outputPanel', 'block');
    
    if (latestOriginalText && latestOriginalText.length > 0) {
      // For lossy compression, compare length only
      if (currentQuality > 0) {
        const originalLen = latestOriginalText.length;
        const decodedLen = decoded.length;
        const lengthDiff = Math.abs(originalLen - decodedLen);
        
        if (lengthDiff <= originalLen * 0.05) { // Within 5%
          safeSetText('outputStatus', '✓ Lossy: within 5% of original');
          const status = document.getElementById('outputStatus');
          if (status) status.className = 'status-badge success';
          showToast('✓ Lossy decompression successful', 'success');
        } else {
          safeSetText('outputStatus', `⚠ Lossy: ${lengthDiff} chars difference`);
          const status = document.getElementById('outputStatus');
          if (status) status.className = 'status-badge warning';
          showToast('⚠ Significant lossy difference detected', 'warning');
        }
      } else if (decoded === latestOriginalText) {
        safeSetText('outputStatus', '✓ Perfect match');
        const status = document.getElementById('outputStatus');
        if (status) status.className = 'status-badge success';
        showToast('✓ Perfect lossless decompression!', 'success');
      } else {
        safeSetText('outputStatus', '⚠ Mismatch detected');
        const status = document.getElementById('outputStatus');
        if (status) status.className = 'status-badge warning';
        showToast('⚠ Decoded text differs from original', 'warning');
      }
    } else {
      safeSetText('outputStatus', '✓ Decompressed');
      const status = document.getElementById('outputStatus');
      if (status) status.className = 'status-badge info';
      showToast('✓ Decompression complete', 'success');
    }
    
  } catch (e) {
    console.error('Decode error:', e);
    showToast(`❌ Decompression failed: ${e.message || e}`, 'error');
  } finally {
    const btn = document.getElementById('btnDecodeAll');
    if (btn) { btn.disabled = false; btn.textContent = '✓ Decompress & Verify'; }
  }
}

async function handleDecodePrefix() {
  try {
    if (!wasmReady || !latestStreamBytes) {
      showToast('No compressed data available', 'warning');
      return;
    }
    
    const upto = Math.max(0, Number(safeGetValue('chunkToDecode')) || 0);
    
    if (upto === 0) {
      showToast('Please select at least 1 chunk to decode', 'warning');
      return;
    }
    
    const btn = document.getElementById('btnDecodePrefix');
    if (btn) { btn.disabled = true; btn.textContent = 'Decoding...'; }
    
    const decoded = decode_prefix_wasm(latestStreamBytes, upto);
    safeSetText('decodedOutput', decoded);
    safeSetDisplay('outputPanel', 'block');
    safeSetText('outputStatus', `Partial decode (${upto} chunk${upto !== 1 ? 's' : ''})`);
    const status = document.getElementById('outputStatus');
    if (status) status.className = 'status-badge info';
    showToast(`✓ Decoded first ${upto} chunk${upto !== 1 ? 's' : ''}`, 'success');
  } catch (e) {
    console.error('Decode prefix error:', e);
    showToast(`❌ Partial decompression failed: ${e.message || e}`, 'error');
  } finally {
    const btn = document.getElementById('btnDecodePrefix');
    if (btn) { btn.disabled = false; btn.textContent = '▶ Decode Partial'; }
  }
}

async function handleCopyOutput() {
  try {
    const el = document.getElementById('decodedOutput');
    const text = el ? (el.textContent || '') : '';
    await navigator.clipboard.writeText(text);
    showToast('✓ Copied to clipboard', 'success');
  } catch (e) {
    console.error('Copy error:', e);
    showToast('❌ Failed to copy', 'error');
  }
}

async function initWasm() {
  try {
    await init();
    wasmReady = true;
    showToast('✓ Ready - Neural & Lossy features enabled', 'success');
    return true;
  } catch (err) {
    console.error('WASM init failed:', err);
    showToast('❌ Failed to load module', 'error');
    return false;
  }
}

// File upload handlers
async function loadBinFile(file) {
  try {
    const ab = await file.arrayBuffer();
    latestStreamBytes = new Uint8Array(ab);
    const chunks = countChunksFromBinary(latestStreamBytes);
    safeSetText('uploadedInfo', `${file.name} — ${file.size.toLocaleString()} bytes`);
    safeSetDisplay('resultsPanel', 'block');
    safeSetDisplay('decodePanel', chunks > 0 ? 'block' : 'none');
    showToast(`✓ Loaded ${file.name}`, 'success');
  } catch (e) {
    showToast(`❌ Failed to load file: ${e}`, 'error');
  }
}

function installFileUploadUI() {
  const fileInput = document.getElementById('fileInput');
  const btnLoadFile = document.getElementById('btnLoadFile');
  const btnDecodeUpload = document.getElementById('btnDecodeUpload');
  const btnClearUpload = document.getElementById('btnClearUpload');
  const uploadArea = document.getElementById('uploadArea');

  if (!fileInput || !btnLoadFile) return;

  btnLoadFile.addEventListener('click', async () => {
    if (fileInput.files && fileInput.files.length > 0) {
      await loadBinFile(fileInput.files[0]);
    }
  });

  if (btnDecodeUpload) {
    btnDecodeUpload.addEventListener('click', handleDecodeAll);
  }

  if (btnClearUpload) {
    btnClearUpload.addEventListener('click', () => {
      latestStreamBytes = null;
      fileInput.value = '';
      safeSetText('uploadedInfo', 'No file loaded');
      hideResults();
      showToast('✓ Cleared', 'info');
    });
  }

  if (uploadArea) {
    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
      uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', async (e) => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        await loadBinFile(e.dataTransfer.files[0]);
      }
    });
  }
}

// Setup event listeners
function setupEventListeners() {
  const inputText = document.getElementById('inputText');
  if (inputText) inputText.addEventListener('input', updateStats);
  
  const btnClear = document.getElementById('btnClear');
  if (btnClear) btnClear.addEventListener('click', handleClear);
  
  const btnEncode = document.getElementById('btnEncode');
  if (btnEncode) btnEncode.addEventListener('click', handleEncode);
  
  const btnDecodeAll = document.getElementById('btnDecodeAll');
  if (btnDecodeAll) btnDecodeAll.addEventListener('click', handleDecodeAll);
  
  const btnDecodePrefix = document.getElementById('btnDecodePrefix');
  if (btnDecodePrefix) btnDecodePrefix.addEventListener('click', handleDecodePrefix);
  
  const btnCopyOutput = document.getElementById('btnCopyOutput');
  if (btnCopyOutput) btnCopyOutput.addEventListener('click', handleCopyOutput);
}

// Initialize
async function initialize() {
  setupEventListeners();
  updateStats();
  installFileUploadUI();
  await initWasm();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}