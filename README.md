# redact-showcase

On-device PII redaction. A transformer model runs **entirely in your browser**
via WebGPU — your text never leaves the tab.

> **Heads up:** the model weights are ~2 GB. The first load downloads them
> once; after that they live in the browser cache and load instantly.

```
 ┌──────────────────────────┐        ┌──────────────────────────┐
 │ Hi, I'm Christian and my │  ───▶  │ Hi, I'm [REDACTED] and   │
 │ email is c@example.com   │        │ my email is [REDACTED]   │
 └──────────────────────────┘        └──────────────────────────┘
```

## How it works

- **Model** — [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter),
  a token-classification transformer fine-tuned to spot PII.
- **Runtime** — [`@huggingface/transformers`](https://huggingface.co/docs/transformers.js/)
  loads the model as quantized ONNX (`q4`) and executes it on your GPU via WebGPU.
- **Caching** — weights are stored in the browser's Cache Storage; subsequent
  visits skip the download entirely.
- **Privacy** — no server, no API call, no telemetry. View Network tab to verify.

## Run locally

```bash
npm install
npm run dev
```

Then open the printed localhost URL. First click on **Load model** triggers the
one-time ~2 GB weight download (tracked with a per-file progress bar). After
that, weights come from the browser cache — zero network on subsequent visits.

## Stack

`Vite` · `TypeScript` · `Transformers.js` · `WebGPU` · `ONNX Runtime Web`