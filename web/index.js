// Plain JS module — no bundler required
// Expects wasm-bindgen output at ./pkg/indic_ans_compressor.js
import init, { encode_stream_wasm, decode_prefix_wasm } from './pkg/indic_ans_compressor.js';

const inputText = document.getElementById('inputText');
const chunkSizeEl = document.getElementById('chunkSize');
const btnEncode = document.getElementById('btnEncode');
const btnDownload = document.getElementById('btnDownload');
const chunkToDecode = document.getElementById('chunkToDecode');
const btnDecodePrefix = document.getElementById('btnDecodePrefix');
const decodedOutput = document.getElementById('decodedOutput');
const streamInfo = document.getElementById('streamInfo');
const btnCountChunks = document.getElementById('btnCountChunks');

let wasmReady = init(); // initialize wasm; returns Promise
let latestStreamBytes = null;

function arrayBufferToHex(ab) {
  const u8 = new Uint8Array(ab);
  return Array.from(u8).map(b => b.toString(16).padStart(2,'0')).join('');
}

// helper: count chunks in binary stream per framing (magic "IC")
function countChunksFromBinary(u8) {
  let pos = 0;
  let count = 0;
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  while (pos + 2 <= u8.length) {
    const m0 = u8[pos], m1 = u8[pos+1];
    if (m0 !== 0x49 || m1 !== 0x43) break; // not "IC"
    pos += 2;
    if (pos + 4 > u8.length) break;
    const token_count = dv.getUint32(pos, false); pos += 4;
    if (pos + 2 > u8.length) break;
    const delta_count = dv.getUint16(pos, false); pos += 2;
    for (let i=0; i<delta_count; i++) {
      if (pos + 4 + 2 > u8.length) { pos = u8.length; break; }
      pos += 4; // id
      const len = dv.getUint16(pos, false); pos += 2;
      pos += len;
      if (pos > u8.length) { pos = u8.length; break; }
    }
    if (pos + 4 > u8.length) break;
    const payload_len = dv.getUint32(pos, false); pos += 4;
    pos += payload_len;
    count++;
  }
  return count;
}

btnEncode.addEventListener('click', async () => {
  await wasmReady;
  const text = inputText.value;
  const chunkSize = Number(chunkSizeEl.value) || 40;
  try {
    const out = encode_stream_wasm(text, chunkSize);
    // ensure a Uint8Array
    const u8 = out instanceof Uint8Array ? out : new Uint8Array(out);
    latestStreamBytes = u8;
    streamInfo.textContent = `Encoded stream: ${u8.length} bytes.`;
    btnDownload.disabled = false;
    btnDecodePrefix.disabled = false;
    btnCountChunks.disabled = false;

    btnDownload.onclick = () => {
      const blob = new Blob([u8], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'indic_stream.bin';
      a.click();
      URL.revokeObjectURL(url);
    };

    const chunks = countChunksFromBinary(u8);
    streamInfo.textContent += ` Chunks: ${chunks}.`;
  } catch (e) {
    console.error('Encode error', e);
    alert('Encode failed: ' + String(e));
  }
});

btnDecodePrefix.addEventListener('click', async () => {
  await wasmReady;
  if (!latestStreamBytes) {
    alert('No stream available. Encode first.');
    return;
  }
  const upto = Math.max(0, Number(chunkToDecode.value) || 0);
  try {
    const result = decode_prefix_wasm(latestStreamBytes, upto);
    decodedOutput.textContent = result || '—';
  } catch (e) {
    console.error('Decode error', e);
    alert('Decode failed: ' + String(e));
  }
});

btnCountChunks.addEventListener('click', () => {
  if (!latestStreamBytes) { alert('No stream available. Encode first.'); return; }
  const count = countChunksFromBinary(latestStreamBytes);
  alert(`Chunks in stream: ${count}`);
});
