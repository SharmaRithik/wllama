import WasmFromPackage from '@reeselevine/wllama-webgpu/esm/wasm-from-package.js';
import wllamaPackageJson from '@reeselevine/wllama-webgpu/package.json';
import { InferenceParams } from './utils/types';

export const WLLAMA_VERSION = wllamaPackageJson.version;

export const WLLAMA_CONFIG_PATHS = WasmFromPackage;

export const MAX_GGUF_SIZE = 2 * 1024 * 1024 * 1024; // 2GB

export interface ModelEntry {
  url: string;
  size: number;
  /** Optional multimodal projector GGUF for vision support. */
  mmprojUrl?: string;
  /** Approximate mmproj file size in bytes (informational). */
  mmprojSize?: number;
}

export const LIST_MODELS: ModelEntry[] = [
  // Vision: SmolVLM-256M (smallest verified vision pair — ideal for smoke-test)
  {
    url: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf',
    size: 175054528,
    mmprojUrl:
      'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf',
    mmprojSize: 103769856,
  },
  // Vision: SmolVLM-2.2B (Q4_K_M LLM + Q8_0 mmproj — same prompt format as 256M)
  {
    url: 'https://huggingface.co/ggml-org/SmolVLM-Instruct-GGUF/resolve/main/SmolVLM-Instruct-Q4_K_M.gguf',
    size: 1112242368,
    mmprojUrl:
      'https://huggingface.co/ggml-org/SmolVLM-Instruct-GGUF/resolve/main/mmproj-SmolVLM-Instruct-Q8_0.gguf',
    mmprojSize: 592521344,
  },
  // Vision: LiquidAI LFM2.5-VL-450M (F16 LLM + F16 mmproj)
  {
    url: 'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-F16.gguf',
    size: 711486624,
    mmprojUrl:
      'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf',
    mmprojSize: 189126080,
  },
  // Vision: Gemma 3 4B IT (higher quality, ~3.3 GB total — uses 64-bit wasm)
  {
    url: 'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf',
    size: 2489894016,
    mmprojUrl:
      'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/mmproj-F16.gguf',
    mmprojSize: 851251328,
  },
  {
    url: 'https://huggingface.co/QuantFactory/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct.Q4_K_M.gguf',
    size: 284164096,
  },
  {
    url: 'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q4_K_M.gguf',
    size: 260046848,
  },
  {
    url: 'https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf',
    size: 279969792,
  },
  {
    url: 'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-F16.gguf',
    size: 542835488,
  },
  {
    url: 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf',
    size: 639447744,
  },
  {
    url: 'https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf',
    size: 766509056,
  },
  {
    url: 'https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf',
    size: 801112064,
  },
  {
    url: 'https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    size: 872415232,
  },
  {
    url: 'https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf',
    size: 1290000000,
  },
  {
    url: 'https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q2_K.gguf',
    size: 1682635744,
  },
  {
    url: 'https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q3_K_M.gguf',
    size: 1795552544,
  },
  {
    url: 'https://huggingface.co/unsloth/SmolLM3-3B-128K-GGUF/resolve/main/SmolLM3-3B-128K-Q4_K_S.gguf',
    size: 1820000000,
  },
  {
    url: 'https://huggingface.co/reeselevine/wllama-split-models/resolve/main/gemma-4-E2B-it-Q4_K_M-00001-of-00005.gguf',
    size: 3110000000,
  },
];

export const DEFAULT_INFERENCE_PARAMS: InferenceParams = {
  nThreads: -1, // auto
  nContext: 4096,
  nPredict: 4096,
  nBatch: 512,
  temperature: 0.2,
  backend: 'webgpu',
};

export const DEFAULT_CHAT_TEMPLATE =
  "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
