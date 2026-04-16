# Miniloader

> **This repository is a public demo.** The full-featured release is available to early access members at [miniloader.ai/test-group-signup](https://miniloader.ai/test-group-signup).

The code in this repo was not fully tested, use at your own risk! It is recommended not to try and build this yourself but to use the build hosted on our website instead.

Miniloader is a desktop application for running a personal local AI stack entirely on your own hardware — no telemetry, no cloud dependencies by default, and no models downloaded at runtime without your explicit action.

---

## What is it?

Miniloader uses a **patchbay metaphor** borrowed from hardware signal routing: every AI capability is a discrete module card, and you connect them by patching virtual cables between typed ports. The result is a composable, inspectable, and reconfigurable AI rack you can see and control.

Under the hood, an asyncio **Hypervisor** manages a directed acyclic graph (DAG) of modules connected by typed wires. Computationally heavy modules run in isolated worker processes so a native crash (e.g. inside llama.cpp) never brings down the rest of the stack.

---

## Modules in this demo

| Module | Description |
|---|---|
| **Local Brain** (`basic_brain`) | Loads `.gguf` model files via `llama-cpp-python` and runs local inference. Supports CPU, Vulkan (AMD/Intel GPU), and CUDA backends. Runs in an isolated worker process. |
| **AI Server** (`gpt_server`) | Exposes the local model as an OpenAI-compatible HTTP API (`/v1/chat/completions`) so any OpenAI client can connect locally. |
| **Chat Terminal** (`gpt_terminal`) | In-app chat interface wired to the AI Server. Supports streaming, system prompts, and tool calling. |
| **Agent Engine** (`agent_engine`) | Agentic orchestration layer — routes tool calls from the model to registered tool providers and feeds results back. |
| **File Vault** (`file_access`) | Grants the agent read/write access to a configured local directory. |
| **Database** (`database`) | Encrypted local SQLite storage for chat history and structured data. |
| **Discord Terminal** (`discord_terminal`) | Connects a Discord bot to the local AI stack as an alternative chat interface. |

### What's in the full release

The full release at [miniloader.ai](https://miniloader.ai/test-group-signup) includes additional modules not present in this demo:

- **Knowledge Engine** — local vector/RAG search over your documents
- **Cloud Brain** — cloud AI provider proxy (OpenAI, Anthropic, etc.) as a drop-in swap for the local model
- **Web Gateway** — secure tunnel for remote access to your local stack
- **Voice** — local speech-to-text and text-to-speech pipeline
- **Web Browser** — Playwright-backed browser tool for the agent
- **Google Suite** and **Obsidian Suite** integrations

---

## Quick start

### Requirements

- Python 3.11
- Windows (primary), Linux (supported)
- A `.gguf` model file (e.g. from [Hugging Face](https://huggingface.co))

### Install

```bash
git clone https://github.com/miniloader/miniloader.git
cd miniloader
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **GPU acceleration:** The default `llama-cpp-python` in `requirements.txt` targets Vulkan. For CUDA or CPU-only builds, follow the [llama-cpp-python installation guide](https://github.com/abetlen/llama-cpp-python#installation) and install the appropriate wheel manually before running `pip install -r requirements.txt`.

### Run

```bash
python main.py
```

On first launch, Miniloader opens a blank rack. Add modules from the right-click menu, connect their ports with cables, and configure each card's settings before starting.


## License

Miniloader Demo is **source-available, not open source.** See [LICENSE.MD](LICENSE.MD) for full terms. Future releases will be open source.

**Short version:**
- You may use, run, and modify Miniloader for personal, academic, or hobbyist purposes.
- Commercial use, redistribution, bundling, and hardware pre-installs require explicit written permission.
- For commercial licensing or partnership inquiries: `chris@kellogg.io` or [miniloader.ai](https://miniloader.ai)

---

## Third-party notices

Third-party software included in or distributed with Miniloader is listed in [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md).

---

## Full release & early access

This repository contains a stripped demo. The full release with all modules, auto-updater, built installer, and cloud features is available to early access testers:

**[miniloader.ai/test-group-signup](https://miniloader.ai/test-group-signup)**
