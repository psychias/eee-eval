# EEE Pipeline — Copilot Agent Instructions

You are the autonomous operator of this LLM evaluation extraction pipeline.
When asked to run the experiments, you take full ownership: you run every
stage, read every output, fix every error, evaluate the results, and iterate
until the pipeline succeeds end-to-end. You never ask the user to do something
you can do yourself.

---

## How to run the pipeline

Use the VSCode terminal to execute tasks in this exact sequence.
Each command activates the venv and loads the .env file first.

### Activation preamble (prepend to every command)
```powershell
.venv\src\Activate.ps1
if (Test-Path .env) {
    Get-Content .env | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } |
    ForEach-Object { $k,$v = $_ -split '=',2;
    [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process') }
}
```

### Stage 0 — Setup
```powershell
python -m venv .venv
# then preamble, then:
pip install -q docling huggingface_hub openai requests jsonschema pandas matplotlib seaborn scipy
```

### Stage 1 — Extract (test batch)
```powershell
python src/extraction/extract_paper.py --paper-list papers/general_llm_papers_batch1.txt --yes
```
This takes several minutes per paper. Watch for:
- `results tables accepted:` — should be > 0 for each paper
- `llm data points:` — should be > 0 if OPENROUTER_API_KEY is set
- `total data points:` — your extraction yield per paper
- `schema validation: N passed, 0 failed` — must be 0 failed

### Stage 2 — Validate
```powershell
python src/validation/validate_outputs.py
```
Must exit with code 0. If any files fail, read them and fix.

### Stage 3 — Aggregate
```powershell
python src/analysis/aggregate_results.py
```
Produces `data/aggregated/all_results.csv` and `coverage_stats.json`.
Read the metadata documentation rates printed at the end.

### Stage 4 — Generate figures
```powershell
python LLM_Evaluation_Report/figures/scripts/coverage_bars.py   # Figure 6 (§4.3)
python LLM_Evaluation_Report/figures/scripts/unified.py         # Figure 5 (§5)
python LLM_Evaluation_Report/figures/scripts/case1.py           # Figure 4 (§5.1)
```
Produces the paper figures in `LLM_Evaluation_Report/figures/`.

---

## Error handling — what to do for each failure type

### ImportError / ModuleNotFoundError
```powershell
pip install <missing_package>
```
Then re-run the failed stage.

### Docling SSL / HuggingFace network error
```powershell
$env:HF_HUB_OFFLINE = "1"
```
The models are already cached at `~/.cache/huggingface/hub/models--docling-project--docling-layout-heron`.
Set the env var and re-run.

### OPENROUTER_API_KEY not set
Read `.env` in the project root. If the key is missing or placeholder:
- Tell the user the key is missing and ask them to paste it
- Write it to `.env` once provided
- Then re-run from Stage 1

### Schema validation failures
Read the failing JSON file. Compare its structure against `eval.schema.json`.
Common causes:
- `evaluator_relationship` value not in the enum → fix in `eval_types.py`
- Missing required field → trace back to `PaperConverter._convert_model`
- Type mismatch on `score` (string instead of float) → fix in `aggregate_results.py`

### Zero data points extracted for a paper
This means both LLM and Docling found nothing. Steps:
1. Check if the PDF downloaded: `src/scrapers/raw/papers/<arxiv_id>.pdf`
2. Run with `--no-llm` flag to test Docling alone
3. Check if the paper has machine-readable text (not scanned):
   ```powershell
   python -c "from docling.document_converter import DocumentConverter; r = DocumentConverter().convert('src/scrapers/raw/papers/<id>.pdf'); print(r.document.export_to_markdown()[:500])"
   ```
4. If scanned PDF: Docling needs OCR enabled. Edit `DoclingParser.__init__` in
   `src/extraction/extract_paper.py` and set `pipeline_options.do_ocr = True`

### LLM rate limit (429)
The pipeline already retries with exponential backoff and model fallbacks.
If all models are exhausted, wait 60 seconds and re-run. The LLM cache means
already-processed chunks are free.

### Figures fail with "no collision pairs found"
This means only one source per model×benchmark — not enough for delta analysis.
This is expected on a 3-paper test run. The figures will skip those panels
and produce what they can. This is not an error.

---

## How to evaluate extraction quality

After Stage 3, read `data/aggregated/all_results.csv` and evaluate:

### Yield check
```powershell
python -c "
import pandas as pd
df = pd.read_csv('data/aggregated/all_results.csv')
print('Total records:', len(df))
print('Models:', df['model'].nunique())
print('Benchmarks:', df['benchmark'].nunique())
print('Papers:', df['source'].nunique())
print()
print('Records per paper:')
print(df.groupby('source')['benchmark'].count().to_string())
"
```
Expected for the 3 test papers:
- `2407.21783` (Llama 3): 50–200 records — large paper with many models
- `2310.06825v1` (Mistral 7B): 20–80 records
- `2505.09388v1`: depends on paper content

If a paper yields 0 records, treat it as a failure and debug.

### Coverage check
```powershell
python -c "
import pandas as pd
df = pd.read_csv('data/aggregated/all_results.csv')
fields = ['shots', 'temperature', 'prompt_template', 'harness', 'chain_of_thought']
for f in fields:
    filled = df[f].notna() & (df[f].astype(str).str.strip() != '')
    pct = filled.sum() / len(df) * 100
    print(f'{f:20s} {filled.sum():4d}/{len(df)} ({pct:.1f}%)')
"
```
Key threshold: `harness` coverage < 20% is a problem — it means the eval
framework was not detected. Check `_detect_eval_library` output in the
extraction log.

### Score sanity check
```powershell
python -c "
import pandas as pd
df = pd.read_csv('data/aggregated/all_results.csv')
df['score'] = pd.to_numeric(df['score'], errors='coerce')
print('Score range: %.4f – %.4f' % (df['score'].min(), df['score'].max()))
print('Scores > 1.0 (should be 0 after normalisation):', (df['score'] > 1.0).sum())
print('Scores == 0.0 (likely empty cells):', (df['score'] == 0.0).sum())
print()
print('Top 10 benchmarks by record count:')
print(df['benchmark'].value_counts().head(10).to_string())
"
```
All scores must be in [0, 1] (except MT-Bench 1–10 and Chatbot Arena 0–3000).
If scores > 1.0 remain, the normalisation in `PaperConverter._convert_model`
has a bug — read that function and fix it.

---

## Iteration rules

- **Never stop after one failure.** Fix the error and re-run the stage.
- **Never re-run a stage that already succeeded** unless a fix requires it.
- **Maximum 3 fix attempts per stage.** If a stage fails 3 times with
  different errors, report what you tried and ask the user for guidance.
- **LLM cache is your friend.** Already-processed chunks cost nothing to
  re-run — the cache in `src/scrapers/raw/llm_cache/` is always used.
- **After fixing a script**, always show the diff of what you changed and why.

---

## Final report format

When all stages succeed, produce a report in this format:

```
═══ EEE Experiment Report ══════════════════════════════════════

Papers processed:    3
Total records:       <N>
Unique models:       <N>
Unique benchmarks:   <N>

Per-paper results:
  2407.21783   (Llama 3)     <N> records  <N> models  <N> benchmarks
  2310.06825v1 (Mistral 7B)  <N> records  <N> models  <N> benchmarks
  2505.09388v1               <N> records  <N> models  <N> benchmarks

Metadata coverage:
  shots            XX%
  temperature      XX%
  prompt_template  XX%
  harness          XX%

Validation:  <N> passed / 0 failed
Figures:     data/figures/ (5 files)

Issues encountered and fixed:
  - <description of any error fixed>

Issues not resolved:
  - <anything that needs human attention>
```

---

## Triggering the agent

To start the autonomous run, say any of:
- "Run the experiments"
- "Run the test batch"
- "Run the full pipeline on arxiv_ids_test.txt"

Claude will then execute all stages autonomously, fix errors, evaluate
results, and produce the report above without further prompting.
