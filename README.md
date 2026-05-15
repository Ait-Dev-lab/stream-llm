# Stream LLM

Streaming LLM inference with GPU shards -- split, upload, and run LLMs from the browser.

## Architecture

```
Colab -> Download GGUF -> Split into per-layer .bin shards -> Upload to CDN
Server (jumpstart) -> Express + model config -> Deploy on Render
Browser (WebGPU) -> fetch -> upload -> compute -> evict -> Real-time streaming
```

## Live Services

| Component | Status | URL |
|-----------|--------|-----|
| Web App | Live | https://ait-dev-lab.github.io/stream-llm/ |
| Jumpstart Server | Live | https://stream-llm-gvta.onrender.com |
| Model CDN | Live | https://github.com/Ait-Dev-lab/stream-llm/releases/tag/v1.0.0 |
| Organization | Live | https://github.com/Ait-Dev-lab |

## Model

- Model: Llama-3.2-3B-Instruct Q4_K_M
- Total layers: 28
- Total shards: 30
- Total size: 2.01 GB

## How It Works

1. User opens web app in Chrome/Edge (WebGPU required)
2. Browser calls jumpstart server for first 8 tokens (instant)
3. Jumpstart returns initial tokens + model config
4. Browser fetches remaining shards from CDN on demand
5. WebGPU processes layers 2-27 on user's GPU
6. Final tokens generated client-side
7. UptimeRobot keeps jumpstart server warm 24/7

## Project Structure

```
stream-llm/
  colab/           -> split_shards.py (GGUF -> .bin shards)
  server/          -> index.js, package.json (jumpstart server)
  browser/         -> index.html, shard-manager.js (WebGPU client)
  shards/          -> model_config.json, tokenizer_config.json
  index.html       -> web app entry point
  shard-manager.js -> StreamWeightManager core
```

## Setup

### Prerequisites

- Google Colab account (for shard generation)
- Render account (free tier)
- GitHub account
- UptimeRobot account (free)

### Steps to Reproduce

1. Run colab/split_shards.py in Colab to generate shards
2. Upload shards to GitHub Releases as CDN
3. Deploy server/ to Render with CDN_URL env var
4. Set up UptimeRobot to ping /health endpoint
5. Enable GitHub Pages on main branch

## Requirements

- Chrome 113+ or Edge 113+ (WebGPU support)
- GPU recommended for fast inference
- Node.js 18+ (for local server testing)

## License

MIT
