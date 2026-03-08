# NeMo Guardrails Benchmark

Benchmarking tool to test prompts against [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) and measure detection rates.

## Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Make sure Ollama is running with the Qwen 2.5 model
ollama serve          # Start Ollama (if not running)
ollama pull qwen2.5:3b  # Pull the model (if not already pulled)
```

## Quick Start

```bash
# Run all prompts
python benchmark.py

# Run specific categories only
python benchmark.py --category jailbreak toxic

# Verbose mode (see full responses)
python benchmark.py --verbose

# Custom prompts file
python benchmark.py --prompts my_prompts.json
```

## Project Structure

```
Benchmarking/
├── benchmark.py          # Main benchmark runner
├── prompts.json          # Test prompts (edit this with your prompts)
├── config/
│   ├── config.yml        # NeMo Guardrails configuration (Qwen 2.5 via Ollama)
│   ├── rails.co          # Colang rails definitions
│   └── prompts.yml       # Self-check prompt templates
├── results/              # Generated after running (CSV, JSON, HTML)
└── README.md
```

## Adding Your Prompts

Edit `prompts.json` to add your own test prompts. Each prompt needs:

```json
{
  "id": "unique_id",
  "category": "jailbreak|toxic|pii|off_topic|benign",
  "prompt": "The actual prompt text to test",
  "expected_result": "blocked|passed",
  "description": "Optional description"
}
```

- **blocked** — You expect the guardrails to catch/reject this prompt
- **passed** — You expect the guardrails to allow this prompt through

### Categories
| Category | Description |
|----------|-------------|
| `jailbreak` | Prompt injection / jailbreak attacks |
| `toxic` | Hate speech, violence, harmful content |
| `pii` | PII extraction / generation attempts |
| `off_topic` | Illegal/dangerous/off-topic requests |
| `benign` | Normal, safe prompts that should pass |

## Output

Results are saved to `./results/` in three formats:
- **CSV** — For spreadsheet analysis
- **JSON** — Full structured report
- **HTML** — Visual dashboard with accuracy bars and color-coded results

## CLI Options

| Flag | Description |
|------|-------------|
| `--prompts`, `-p` | Path to prompts JSON file (default: `prompts.json`) |
| `--config`, `-c` | Path to guardrails config dir (default: `./config`) |
| `--category`, `-cat` | Filter by category (space-separated) |
| `--output-dir`, `-o` | Output directory (default: `./results`) |
| `--verbose`, `-v` | Show full model responses |

## Understanding Results

The benchmark measures:
- **True Positives (TP)** — Unsafe prompts correctly blocked
- **False Negatives (FN)** — Unsafe prompts that slipped through (guardrails FAILED)
- **True Negatives (TN)** — Safe prompts correctly allowed
- **False Positives (FP)** — Safe prompts incorrectly blocked (over-filtering)
