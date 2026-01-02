// Enhanced Indic Compressor Web Interface
import init, { encode_stream_wasm, decode_prefix_wasm, decode_full_wasm } from './pkg/indic_ans_compressor.js';

// DOM Elements
const inputText = document.getElementById('inputText');
const chunkSizeEl = document.getElementById('chunkSize');
const chunkSizeDisplay = document.getElementById('chunkSizeDisplay');
const chunkToDecode = document.getElementById('chunkToDecode');
const chunkToDecodeDisplay = document.getElementById('chunkToDecodeDisplay');
const charCount = document.getElementById('charCount');
const byteCount = document.getElementById('byteCount');

const btnEncode = document.getElementById('btnEncode');
const btnClear = document.getElementById('btnClear');
const btnDownload = document.getElementById('btnDownload');
const btnDecodeAll = document.getElementById('btnDecodeAll');
const btnDecodePrefix = document.getElementById('btnDecodePrefix');
const btnCopyOutput = document.getElementById('btnCopyOutput');

const resultsPanel = document.getElementById('resultsPanel');
const decodePanel = document.getElementById('decodePanel');
const outputPanel = document.getElementById('outputPanel');

const originalSize = document.getElementById('originalSize');
const compressedSize = document.getElementById('compressedSize');
const compressionRatio = document.getElementById('compressionRatio');
const chunkCount = document.getElementById('chunkCount');

const decodedOutput = document.getElementById('decodedOutput');
const outputStatus = document.getElementById('outputStatus');
const statusToast = document.getElementById('statusToast');
const toastMessage = document.getElementById('toastMessage');

// State
let wasmReady = init();
let latestStreamBytes = null;
let latestOriginalText = '';

// Initialize
updateStats();

// Event Listeners
inputText.addEventListener('input', updateStats);

chunkSizeEl.addEventListener('input', () => {
  chunkSizeDisplay.textContent = chunkSizeEl.value;
});

chunkToDecode.addEventListener('input', () => {
  chunkToDecodeDisplay.textContent = chunkToDecode.value;
});

btnClear.addEventListener('click', () => {
  inputText.value = '';
  updateStats();
  hideResults();
  showToast('Cleared', 'success');
});

btnEncode.addEventListener('click', handleEncode);
btnDecodeAll.addEventListener('click', handleDecodeAll);
btnDecodePrefix.addEventListener('click', handleDecodePrefix);
btnCopyOutput.addEventListener('click', handleCopyOutput);

// Functions
function updateStats() {
  const text = inputText.value;
  const chars = text.length;
  const bytes = new TextEncoder().encode(text).length;
  
  charCount.textContent = chars.toLocaleString();
  byteCount.textContent = bytes.toLocaleString();
}

function hideResults() {
  resultsPanel.style.display = 'none';
  decodePanel.style.display = 'none';
  outputPanel.style.display = 'none';
  latestStreamBytes = null;
}

function showToast(message, type = 'info') {
  toastMessage.textContent = message;
  statusToast.className = `toast ${type}`;
  statusToast.classList.remove('hidden');
  
  setTimeout(() => {
    statusToast.classList.add('hidden');
  }, 3000);
}

function countChunksFromBinary(u8) {
  let pos = 0;
  let count = 0;
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  
  try {
    while (pos + 2 <= u8.length) {
      const m0 = u8[pos], m1 = u8[pos+1];
      if (m0 !== 0x49 || m1 !== 0x43) break; // not "IC"
      pos += 2;
      
      if (pos + 4 > u8.length) break;
      const token_count = dv.getUint32(pos, false); pos += 4;
      
      if (pos + 2 > u8.length) break;
      const delta_count = dv.getUint16(pos, false); pos += 2;
      
      // Skip deltas
      for (let i = 0; i < delta_count; i++) {
        if (pos + 6 > u8.length) { pos = u8.length; break; }
        pos += 4; // id
        const len = dv.getUint16(pos, false); pos += 2;
        pos += len;
        if (pos > u8.length) { pos = u8.length; break; }
      }
      
      // Skip features
      pos += token_count;
      if (pos > u8.length) break;
      
      // Skip payload
      if (pos + 4 > u8.length) break;
      const payload_len = dv.getUint32(pos, false); pos += 4;
      pos += payload_len;
      
      count++;
    }
  } catch (e) {
    console.error('Error counting chunks:', e);
  }
  
  return count;
}

async function handleEncode() {
  try {
    await wasmReady;
    
    const text = inputText.value.trim();
    if (!text) {
      showToast('Please enter some text to compress', 'warning');
      return;
    }
    
    latestOriginalText = text;
    const originalBytes = new TextEncoder().encode(text).length;
    const chunkSize = Number(chunkSizeEl.value) || 40;
    
    // Show loading state
    btnEncode.disabled = true;
    btnEncode.textContent = 'Compressing...';
    
    // Encode
    const encoded = encode_stream_wasm(text, chunkSize);
    const u8 = encoded instanceof Uint8Array ? encoded : new Uint8Array(encoded);
    latestStreamBytes = u8;
    
    const compressedBytes = u8.length;
    const ratio = (compressedBytes / originalBytes * 100).toFixed(1);
    const chunks = countChunksFromBinary(u8);
    
    // Update results
    originalSize.textContent = `${originalBytes.toLocaleString()} bytes`;
    compressedSize.textContent = `${compressedBytes.toLocaleString()} bytes`;
    compressionRatio.textContent = `${ratio}% (${(originalBytes - compressedBytes).toLocaleString()} bytes saved)`;
    chunkCount.textContent = chunks.toLocaleString();
    
    // Setup progressive decode slider
    chunkToDecode.max = chunks;
    chunkToDecode.value = chunks;
    chunkToDecodeDisplay.textContent = chunks;
    
    // Show results
    resultsPanel.style.display = 'block';
    decodePanel.style.display = 'block';
    
    // Setup download
    btnDownload.onclick = () => {
      const blob = new Blob([u8], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'indic_compressed.bin';
      a.click();
      URL.revokeObjectURL(url);
      showToast('File downloaded', 'success');
    };
    
    showToast(`Compressed to ${ratio}% of original size`, 'success');
    
  } catch (e) {
    console.error('Encode error:', e);
    showToast(`Compression failed: ${e.message || e}`, 'error');
  } finally {
    btnEncode.disabled = false;
    btnEncode.textContent = '🗜️ Compress Text';
  }
}

async function handleDecodeAll() {
  try {
    await wasmReady;
    
    if (!latestStreamBytes) {
      showToast('No compressed data available. Compress first.', 'warning');
      return;
    }
    
    btnDecodeAll.disabled = true;
    btnDecodeAll.textContent = 'Decompressing...';
    
    const decoded = decode_full_wasm(latestStreamBytes);
    
    decodedOutput.textContent = decoded;
    outputPanel.style.display = 'block';
    
    // Check if matches original
    if (decoded === latestOriginalText) {
      outputStatus.textContent = 'Perfect match';
      outputStatus.className = 'status-badge success';
      showToast('Decompression successful - perfect match!', 'success');
    } else {
      outputStatus.textContent = 'Mismatch detected';
      outputStatus.className = 'status-badge warning';
      showToast('Decoded text does not match original', 'warning');
    }
    
  } catch (e) {
    console.error('Decode error:', e);
    showToast(`Decompression failed: ${e.message || e}`, 'error');
    decodedOutput.textContent = `Error: ${e.message || e}`;
    outputPanel.style.display = 'block';
    outputStatus.textContent = ' Error';
    outputStatus.className = 'status-badge error';
  } finally {
    btnDecodeAll.disabled = false;
    btnDecodeAll.textContent = ' Decompress All';
  }
}

async function handleDecodePrefix() {
  try {
    await wasmReady;
    
    if (!latestStreamBytes) {
      showToast(' No compressed data available. Compress first.', 'warning');
      return;
    }
    
    const upto = Math.max(0, Number(chunkToDecode.value) || 0);
    
    btnDecodePrefix.disabled = true;
    btnDecodePrefix.textContent = ' Decoding...';
    
    const decoded = decode_prefix_wasm(latestStreamBytes, upto);
    
    decodedOutput.textContent = decoded;
    outputPanel.style.display = 'block';
    
    outputStatus.textContent = ` Partial decode (${upto} chunk${upto !== 1 ? 's' : ''})`;
    outputStatus.className = 'status-badge info';
    
    showToast(` Decoded first ${upto} chunk${upto !== 1 ? 's' : ''}`, 'success');
    
  } catch (e) {
    console.error('Decode prefix error:', e);
    showToast(` Partial decompression failed: ${e.message || e}`, 'error');
    decodedOutput.textContent = `Error: ${e.message || e}`;
    outputPanel.style.display = 'block';
    outputStatus.textContent = ' Error';
    outputStatus.className = 'status-badge error';
  } finally {
    btnDecodePrefix.disabled = false;
    btnDecodePrefix.textContent = 'Decode Partial';
  }
}

async function handleCopyOutput() {
  try {
    const text = decodedOutput.textContent;
    await navigator.clipboard.writeText(text);
    showToast(' Copied to clipboard', 'success');
  } catch (e) {
    console.error('Copy error:', e);
    showToast(' Failed to copy', 'error');
  }
}

// Error handling for WASM loading
wasmReady.catch(err => {
  console.error('Failed to initialize WASM:', err);
  showToast(' Failed to load WebAssembly module. Please refresh the page.', 'error');
  btnEncode.disabled = true;
});