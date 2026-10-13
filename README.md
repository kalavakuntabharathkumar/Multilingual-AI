# Multilingual LLM Evaluation and Judge Validation Framework

A YAML-driven evaluation harness comparing an API model and a local Hugging Face model across Hindi, Telugu, and Tamil. It includes prompt execution adapters, human-rating ingestion, weighted Cohen's kappa, rubric comparison, bootstrap confidence intervals, and Markdown reporting.

## Data provenance
The checked-in records are **synthetic demonstration data**, not genuine human ratings or measured model outputs. Replace `data/demo/` with authorized real evaluation data before making empirical claims. The metrics in any resume bullet (300 prompts, 200 ratings, kappa 0.52→0.68, 11-point language gap) should only be claimed if reproduced from genuine runs.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --config configs/demo.yaml --mode demo
pytest -q
```
Report files are written to `outputs/`.

## Live providers
Set `OPENAI_API_KEY`, install `requirements-openai.txt`, and run:
```bash
python -m src.pipeline --config configs/evaluation.yaml --mode live --provider openai
```
For a local Transformers model, install `requirements-transformers.txt` and run:
```bash
python -m src.pipeline --config configs/evaluation.yaml --mode live --provider transformers
```
The default Qwen checkpoint is configurable. For Colab T4, install the Transformers requirements in the notebook and run from the repository directory. GPU memory, model access, licensing, and throughput depend on the selected checkpoint.

## Input schemas
- Prompts: `prompt_id,language,prompt`
- Human ratings: `prompt_id,rater_id,rating` (integer 1–5)
- Judge scores: `prompt_id,language,rubric_version,judge_rating`

Set the seed and bootstrap iterations in YAML. Keep raw input files immutable. Bootstrap intervals quantify sampling uncertainty, not sampling bias.
