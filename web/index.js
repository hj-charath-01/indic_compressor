// Enhanced Indic Compressor Web Interface - Production Version
import init, { encode_stream_wasm, decode_prefix_wasm, decode_full_wasm } from './pkg/indic_ans_compressor.js';

// State
let wasmReady = false;
let latestStreamBytes = null;
let latestOriginalText = '';

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

function safeSetAttribute(id, attr, value) {
  const el = document.getElementById(id);
  if (el) {
    el.setAttribute(attr, value);
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
  } catch (e) {
    console.error('Error updating stats:', e);
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

// Initialize slider displays
(function initSliderDisplays() {
  const chunkSize = document.getElementById('chunkSize');
  const chunkSizeDisplay = document.getElementById('chunkSizeDisplay');
  if (chunkSize && chunkSizeDisplay) {
    chunkSizeDisplay.textContent = chunkSize.value;
    function updateSliderFill(sl) {
      const min = Number(sl.min || 0);
      const max = Number(sl.max || 100);
      const val = Number(sl.value);
      const percent = (val - min) / (max - min) * 100;
      sl.style.background = `linear-gradient(90deg, var(--primary) ${percent}%, var(--border) ${percent}%)`;
    }
    updateSliderFill(chunkSize);
    chunkSize.addEventListener('input', () => {
      chunkSizeDisplay.textContent = chunkSize.value;
      updateSliderFill(chunkSize);
    });
  }

  const chunkToDecode = document.getElementById('chunkToDecode');
  const chunkToDecodeDisplay = document.getElementById('chunkToDecodeDisplay');
  if (chunkToDecode && chunkToDecodeDisplay) {
    chunkToDecodeDisplay.textContent = chunkToDecode.value;
    function updateDecodeFill(sl) {
      const min = Number(sl.min || 0);
      const max = Number(sl.max || 1);
      const val = Number(sl.value);
      const percent = max === min ? 0 : (val - min) / (max - min) * 100;
      sl.style.background = `linear-gradient(90deg, var(--primary) ${percent}%, var(--border) ${percent}%)`;
    }
    updateDecodeFill(chunkToDecode);
    chunkToDecode.addEventListener('input', () => {
      chunkToDecodeDisplay.textContent = chunkToDecode.value;
      updateDecodeFill(chunkToDecode);
    });
  }
})();

/**
 * Count chunks by parsing binary stream format.
 * Format per chunk:
 * - magic "IC" (2 bytes)
 * - token_count (u16 BE)
 * - delta_count (u8)
 * - deltas: [id (u16 BE), token_len (u8), token_bytes]*
 * - features (token_count bytes)
 * - payload_len (u16 BE)
 * - payload bytes
 */
function countChunksFromBinary(u8) {
  if (!u8 || u8.length < 2) return 0;
  
  // Check for uncompressed format
  if (u8[0] === 0x55 && u8[1] === 0x43) return 1; // "UC"
  
  // Check for compressed format
  if (u8[0] !== 0x49 || u8[1] !== 0x43) return 0; // Not "IC"

  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let pos = 0;
  let count = 0;
  
  try {
    while (pos + 2 <= u8.length) {
      // Check magic bytes
      if (u8[pos] !== 0x49 || u8[pos+1] !== 0x43) break;
      pos += 2;
      
      // Read token_count (u16 BE)
      if (pos + 2 > u8.length) break;
      const token_count = dv.getUint16(pos, false);
      pos += 2;
      
      // Read delta_count (u8)
      if (pos + 1 > u8.length) break;
      const delta_count = u8[pos];
      pos += 1;
      
      // Skip deltas
      for (let i = 0; i < delta_count; i++) {
        if (pos + 3 > u8.length) { pos = u8.length; break; }
        pos += 2; // id (u16)
        const len = u8[pos];
        pos += 1; // token_len (u8)
        if (pos + len > u8.length) { pos = u8.length; break; }
        pos += len; // token bytes
      }
      
      // Skip features
      if (pos + token_count > u8.length) break;
      pos += token_count;
      
      // Read payload_len (u16 BE)
      if (pos + 2 > u8.length) break;
      const payload_len = dv.getUint16(pos, false);
      pos += 2;
      
      // Skip payload
      if (pos + payload_len > u8.length) break;
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
  if (u8[0] === 0x55 && u8[1] === 0x43) return 'uncompressed';
  if (u8[0] === 0x49 && u8[1] === 0x43) return 'compressed';
  return 'unknown';
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
    
    const btn = document.getElementById('btnEncode');
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Compressing...';
    }
    
    const encoded = encode_stream_wasm(text, chunkSize);
    const u8 = encoded instanceof Uint8Array ? encoded : new Uint8Array(encoded);
    latestStreamBytes = u8;
    
    const compressedBytes = u8.length;
    const ratio = (compressedBytes / Math.max(1, originalBytes) * 100).toFixed(1);
    const chunks = countChunksFromBinary(u8);
    const format = getCompressionFormat(u8);
    
    const isUncompressed = format === 'uncompressed';
    const savings = originalBytes - compressedBytes;
    
    safeSetText('originalSize', `${originalBytes.toLocaleString()} bytes`);
    safeSetText('compressedSize', `${compressedBytes.toLocaleString()} bytes`);
    
    if (isUncompressed) {
      safeSetHTML('compressionRatio', 
        `<span style="color: #6c757d;">Stored uncompressed (text too small)</span>`);
      showToast('ℹ️ Text stored uncompressed (too small to benefit from compression)', 'info');
    } else if (savings > 0) {
      safeSetText('compressionRatio', `${ratio}% (${savings.toLocaleString()} bytes saved)`);
      showToast(`✓ Compressed to ${ratio}% of original size`, 'success');
    } else {
      safeSetText('compressionRatio', `${ratio}% (${Math.abs(savings).toLocaleString()} bytes overhead)`);
      showToast(`⚠ Result is larger (${ratio}%)`, 'warning');
    }
    
    safeSetText('chunkCount', chunks.toLocaleString());
    
    safeSetAttribute('chunkToDecode', 'max', Math.max(0, chunks));
    safeSetAttribute('chunkToDecode', 'value', Math.max(0, chunks));
    safeSetText('chunkToDecodeDisplay', Math.max(0, chunks));
    
    // Update slider fill
    const chunkToDecodeSlider = document.getElementById('chunkToDecode');
    if (chunkToDecodeSlider) {
      const min = Number(chunkToDecodeSlider.min || 0);
      const max = Number(chunkToDecodeSlider.max || 1);
      const val = Number(chunkToDecodeSlider.value);
      const percent = max === min ? 0 : (val - min) / (max - min) * 100;
      chunkToDecodeSlider.style.background = `linear-gradient(90deg, var(--primary) ${percent}%, var(--border) ${percent}%)`;
    }
    
    safeSetDisplay('resultsPanel', 'block');
    if (!isUncompressed && chunks > 1) {
      safeSetDisplay('decodePanel', 'block');
    } else {
      safeSetDisplay('decodePanel', 'none');
    }
    
    const downloadBtn = document.getElementById('btnDownload');
    if (downloadBtn) {
      downloadBtn.onclick = () => {
        try {
          const blob = new Blob([u8], { type: 'application/octet-stream' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'indic_compressed.bin';
          a.click();
          URL.revokeObjectURL(url);
          showToast('✓ File downloaded', 'success');
        } catch (e) {
          console.error('Download error:', e);
          showToast('❌ Download failed', 'error');
        }
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
    
    const fmt = getCompressionFormat(latestStreamBytes);
    let decoded = '';
    
    if (fmt === 'uncompressed') {
      // Uncompressed format: skip magic (2 bytes) + length (4 bytes)
      if (latestStreamBytes.length >= 6) {
        decoded = new TextDecoder().decode(latestStreamBytes.slice(6));
      } else {
        throw new Error('Invalid uncompressed format');
      }
    } else {
      if (!wasmReady) {
        showToast('WASM not ready for decompression', 'error');
        return;
      }
      decoded = decode_full_wasm(latestStreamBytes);
    }
    
    safeSetText('decodedOutput', decoded);
    safeSetDisplay('outputPanel', 'block');
    
    if (latestOriginalText && latestOriginalText.length > 0) {
      if (decoded === latestOriginalText) {
        safeSetText('outputStatus', '✓ Perfect match');
        const status = document.getElementById('outputStatus');
        if (status) status.className = 'status-badge success';
        showToast('✓ Decompression successful - perfect match!', 'success');
      } else {
        safeSetText('outputStatus', '⚠ Mismatch detected');
        const status = document.getElementById('outputStatus');
        if (status) status.className = 'status-badge warning';
        showToast('⚠ Decoded text does not match original', 'warning');
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
    if (btn) { btn.disabled = false; btn.textContent = '✓ Decompress All'; }
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
    showToast('✓ Ready to compress', 'success');
    return true;
  } catch (err) {
    console.error('WASM init failed:', err);
    showToast('❌ Failed to load module', 'error');
    const btn = document.getElementById('btnEncode');
    if (btn) btn.disabled = true;
    return false;
  }
}

// File upload handlers
async function loadBinFile(file) {
  try {
    if (!file) throw new Error("No file provided");
    const name = file.name || "uploaded.bin";
    const size = file.size || 0;

    const ab = await file.arrayBuffer();
    latestStreamBytes = new Uint8Array(ab);

    safeSetText('uploadedInfo', `${name} — ${size.toLocaleString()} bytes`);

    const fmt = getCompressionFormat(latestStreamBytes);
    const chunks = countChunksFromBinary(latestStreamBytes);
    
    safeSetText('chunkCount', `${chunks}`);
    safeSetAttribute('chunkToDecode', 'max', Math.max(0, chunks));
    safeSetAttribute('chunkToDecode', 'value', Math.max(0, chunks));
    safeSetText('chunkToDecodeDisplay', `${Math.max(0, chunks)}`);

    const chunkToDecodeSlider = document.getElementById('chunkToDecode');
    if (chunkToDecodeSlider) {
      const min = Number(chunkToDecodeSlider.min || 0);
      const max = Number(chunkToDecodeSlider.max || 1);
      const val = Number(chunkToDecodeSlider.value);
      const percent = max === min ? 0 : (val - min) / (max - min) * 100;
      chunkToDecodeSlider.style.background = `linear-gradient(90deg, var(--primary) ${percent}%, var(--border) ${percent}%)`;
    }

    safeSetDisplay('resultsPanel', 'block');
    safeSetDisplay('decodePanel', chunks > 0 ? 'block' : 'none');

    showToast(`✓ Loaded ${name} (${size.toLocaleString()} bytes) — ${fmt}`, 'success');
  } catch (e) {
    console.error("loadBinFile error:", e);
    showToast(`❌ Failed to load file: ${e.message || e}`, 'error');
  }
}

function installFileUploadUI() {
  const fileInput = document.getElementById('fileInput');
  const btnLoadFile = document.getElementById('btnLoadFile');
  const btnDecodeUpload = document.getElementById('btnDecodeUpload');
  const btnClearUpload = document.getElementById('btnClearUpload');
  const uploadArea = document.getElementById('uploadArea');

  if (!fileInput || !btnLoadFile || !btnDecodeUpload || !btnClearUpload || !uploadArea) {
    return;
  }

  btnLoadFile.addEventListener('click', async () => {
    if (!fileInput.files || fileInput.files.length === 0) {
      showToast('Please choose a .bin file first', 'warning');
      return;
    }
    const f = fileInput.files[0];
    await loadBinFile(f);
  });

  btnDecodeUpload.addEventListener('click', async () => {
    if (!latestStreamBytes) {
      showToast('No file loaded to decode', 'warning');
      return;
    }
    await handleDecodeAll();
  });

  btnClearUpload.addEventListener('click', () => {
    latestStreamBytes = null;
    latestOriginalText = '';
    if (fileInput) fileInput.value = '';
    safeSetText('uploadedInfo', 'No file loaded');
    safeSetText('chunkCount', '—');
    safeSetAttribute('chunkToDecode', 'max', 0);
    safeSetAttribute('chunkToDecode', 'value', 0);
    safeSetText('chunkToDecodeDisplay', '0');
    safeSetDisplay('resultsPanel', 'none');
    safeSetDisplay('decodePanel', 'none');
    safeSetDisplay('outputPanel', 'none');
    showToast('✓ Cleared uploaded file', 'info');
  });

  uploadArea.addEventListener('dragover', (ev) => {
    ev.preventDefault();
    uploadArea.classList.add('dragover');
  });
  
  uploadArea.addEventListener('dragleave', (ev) => {
    uploadArea.classList.remove('dragover');
  });
  
  uploadArea.addEventListener('drop', async (ev) => {
    ev.preventDefault();
    uploadArea.classList.remove('dragover');
    const dt = ev.dataTransfer;
    if (!dt || !dt.files || dt.files.length === 0) {
      showToast('No file dropped', 'warning');
      return;
    }
    const f = dt.files[0];
    await loadBinFile(f);
  });
}

// Setup event listeners
function setupEventListeners() {
  const inputText = document.getElementById('inputText');
  if (inputText) inputText.addEventListener('input', updateStats);
  
  const chunkSize = document.getElementById('chunkSize');
  const chunkSizeDisplay = document.getElementById('chunkSizeDisplay');
  if (chunkSize && chunkSizeDisplay) {
    chunkSize.addEventListener('input', () => {
      chunkSizeDisplay.textContent = chunkSize.value;
    });
  }
  
  const chunkToDecode = document.getElementById('chunkToDecode');
  const chunkToDecodeDisplay = document.getElementById('chunkToDecodeDisplay');
  if (chunkToDecode && chunkToDecodeDisplay) {
    chunkToDecode.addEventListener('input', () => {
      chunkToDecodeDisplay.textContent = chunkToDecode.value;
    });
  }
  
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