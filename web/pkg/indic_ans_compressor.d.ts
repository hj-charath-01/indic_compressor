/* tslint:disable */
/* eslint-disable */

export function decode_full_wasm(data: Uint8Array): string;

export function decode_prefix_wasm(data: Uint8Array, upto: number): string;

export function encode_stream_advanced_wasm(text: string, chunk_size: number, use_neural: boolean, neural_weight: number, quality_level: number): Uint8Array;

export function encode_stream_wasm(text: string, chunk_size: number): Uint8Array;

export function estimate_lossy_savings(text: string, quality_level: number): string;

export function get_compression_info(text: string, chunk_size: number): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly decode_full_wasm: (a: number, b: number, c: number) => void;
  readonly decode_prefix_wasm: (a: number, b: number, c: number, d: number) => void;
  readonly encode_stream_advanced_wasm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly encode_stream_wasm: (a: number, b: number, c: number, d: number) => void;
  readonly estimate_lossy_savings: (a: number, b: number, c: number, d: number) => void;
  readonly get_compression_info: (a: number, b: number, c: number, d: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_export: (a: number, b: number) => number;
  readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
