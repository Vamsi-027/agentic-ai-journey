# 🤖 Agentic AI Journey

Welcome to my Agentic AI Mastery learning and R&D repository! This codebase is structured to transition from backend/systems engineering into **AI Systems & Agentic AI Engineering**. 

Instead of scattered scripts, this repository is designed as a **production-ready workspace-based monorepo**, adhering to senior engineering practices: modular layers, reusable helper packages, decoupled prompt templates, and isolated log/database files.

---

## 📂 Repository Architecture

```text
agentic-ai-journey/
├── .env.example                  # Environment variables template
├── .gitignore                    # Global git exclusions (venv, DBs, logs, etc.)
├── pyproject.toml                # Unified package & dependency workspace configuration
├── README.md                     # This portfolio overview and index
├── docs/                         # Study notes, paper deep-dives, and roadmap check-ins
│   ├── roadmap.md                # Interactive 12-month study plan
│   └── papers/                   # Summaries and notes on research papers
├── papers/                       # Raw PDF storage for research papers
├── prompts/                      # Centralized, version-controlled prompt templates registry
│   └── concept_explanations.yaml # System prompts and user templates in YAML
├── src/                          # The core reusable agent infrastructure package
│   ├── core/
│   │   ├── config.py             # Centralized environment settings loader
│   │   ├── client.py             # Instrumented client wrappers (with rate-limiting backoff)
│   │   └── database.py           # Shared SQLite session and logging connector
│   └── telemetry/
│       └── logger.py             # Standard multi-handler logger (stdout + data/app.log)
├── experiments/                  # Scratchpads, notebook ports, and prototype scripts
│   ├── prompt_testing/           # Multi-variant prompt testing harness and scripts
│   └── nanogpt/                  # PyTorch model training code (character-level transformer)
├── projects/                     # End-to-end modular agent projects (Month 2+)
├── evals/                        # Dedicated testing and evaluation harness (Month 5+)
└── data/                         # Persistent local storage (SQLite databases, logs) - Git Ignored
```

---

## ⚡ Getting Started

### Prerequisites
- Python `3.10` or higher.

### 1. Clone & Set Up Credentials
Copy the environment variables template and add your credentials:
```bash
cp .env.example .env
```
Edit the newly created `.env` file and insert your `ANTHROPIC_API_KEY`.

### 2. Set Up Virtual Environment & Dependencies
We use a single virtual environment at the project root to manage dependencies across all experiments and projects:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies and the src/ module in editable mode
pip install -e .
```
> [!TIP]
> Installing in editable mode (`-e .`) registers the `src` folder as a local package, allowing you to run clean imports like `from src.core.client import AgenticClient` in any script without import path errors.

---

## ⚙️ Core Infrastructure Overview

### 1. Centralized Prompt Management (`prompts/`)
System and user prompts are decoupled from Python scripts and stored inside `prompts/concept_explanations.yaml`. This lets you modify or test prompts without touching python code.

### 2. Automated Backoff Retries (`src/core/client.py`)
Our core wrapper `AgenticClient` extends the official Anthropic SDK and integrates the `backoff` package, automatically recovering from `RateLimitError` or temporary API status errors.

### 3. Consolidated Logging & DB Storage (`data/`)
Databases (`prompt_experiments.db`) and file logs (`app.log`) are isolated to a root-level `data/` folder, which is git-ignored to prevent workspace clutter.

---

## 🧪 Running Experiments

### Run Prompt Test Script
Verify basic connectivity and configuration:
```bash
python experiments/prompt_testing/main.py
```

### Run Prompt Experiment Harness
Execute parallel prompt testing iterations (27 runs testing various model, temperature, and prompt combinations):
```bash
python experiments/prompt_testing/harness.py
```
*Outputs are printed as a summary table in the console and persisted in `data/prompt_experiments.db`.*

### Train nanoGPT
Run the Shakespeare bigram language model script:
```bash
python experiments/nanogpt/main.py
```

---

## 🗺️ Learning Roadmap
My active progress tracker is hosted at [docs/roadmap.md](file:///Users/vamsi_cheruku/Desktop/Agentic%20AI%20Journey/docs/roadmap.md). It outlines my 12-month study plan across LLM foundations, agent architectures, multi-agent orchestrations, context engineering, and evaluations.
