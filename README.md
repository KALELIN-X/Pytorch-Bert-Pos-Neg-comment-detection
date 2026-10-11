# CommentScope comment evaluator

A local Python application with two distinct evaluation tasks:

- **Sentiment (new):** enter a sentence or analyze post comments as positive, neutral or negative. Paste comments, upload a CSV, or retrieve a bounded sample through the official YouTube API when a local key is configured. Post percentages count predicted labels among the analyzed comments.
- **Emotions & toxicity (reconstructed):** the original six toxicity labels and six positive emotions, with threshold flags, an explanation and a weighted net-goodness heuristic. This task is different from general sentiment: a comment can have both kinds of signals or neither.

CommentScope is the application's new name. Legacy CivicBalance model identifiers, checkpoint paths and database names remain unchanged for compatibility with the preserved training workflow.

## Repository findings and reconstruction

The original `main` branch contained a short README and a `Training Data` file of dataset links. The closed, unmerged `feat/civicbalance-api` branch at `300702c` contained a shared-BERT/two-head training skeleton, FastAPI, Streamlit, optional Postgres ETL/SQL, and blank report templates. This reconstruction incorporates that branch to preserve its history and useful work.

No training CSVs, trained CivicBalance checkpoint, tests, or measured evaluation results were present. The earlier API could not start without the missing checkpoint. Training had a NumPy scope error, did not accept BERT's `token_type_ids`, silently filled missing emotion labels with zeros, and calculated class weights from validation rows too. Launch paths and dataset documentation were inconsistent.

**Rebuilt or added here:** runnable local inference with pinned public replacement models; strict single/batch API contracts; readiness and safe failure responses; a cleaner dashboard; repaired optional shared-BERT training and portable checkpoint loading; input/error/training tests; reproducible model provenance and installation instructions. Separately requested additions are three-class sentiment, post-comment percentages, manual/CSV import, and optional official YouTube retrieval. These are reconstruction/new changes, not claims about historical implementation or performance. The original report metrics remain unmeasured templates. No historical precision/recall improvement is verified by this repository.

## Architecture and prediction flow

The dashboard defaults to the newly added sentiment task. Its local DistilBERT classifier produces three softmax scores and selects the highest-scoring label. For a post, the application counts those predicted labels and calculates `label_count / analyzed_count * 100`. These percentages describe the analyzed sample; they are not confidence, verified truth, or the share of all comments on a platform. The original emotion/toxicity task remains selectable independently.

```text
Streamlit dashboard or HTTP client
  -> FastAPI: validate JSON, comment lengths, batch size, threshold and weights
  -> local tokenizer(s): reject comments exceeding the token budget
  -> local PyTorch model(s): inference mode, serialized CPU predictions
  -> sentiment: softmax scores -> highest-scoring label -> sample counts/percentages
  -> emotions/toxicity: sigmoid scores -> threshold flags -> explanation + heuristic
```

The default `public` backend uses two independently trained public models because the original custom checkpoint is missing:

- `unitary/toxic-bert`: six toxicity outputs.
- `SamLowe/roberta-base-go_emotions`: 28 emotion outputs; this application selects the six positive emotions already defined in the original project.

The optional `shared_bert` backend loads a checkpoint produced by the repaired original-style trainer: one BERT encoder and two six-output linear heads. Neither backend requires Postgres. Downloading models is an explicit setup step; API prediction uses local files only and does not send comments to a hosted inference service. Submitted comments are not persisted by the application. Run the demo bound to localhost with one API worker; multiple workers duplicate model memory.

| Output group | Labels |
| --- | --- |
| Toxicity | `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` |
| Positive emotions | `admiration`, `gratitude`, `joy`, `pride`, `love`, `optimism` |

New summary labels are `positive_signals`, `toxicity_signals`, `mixed_signals`, and `no_detected_signals`. They describe threshold crossings, not generic sentiment. The original API did not expose a neutral category. Although the public emotion model has a neutral output, this application preserves the original selected label set. No detected signals does not establish neutrality or safety.

Emotion/toxicity scores are **uncalibrated sigmoid outputs**, not confidence percentages. The separate sentiment task uses uncalibrated softmax outputs. Net goodness is the tunable heuristic:

```text
w_pos * mean(six positive emotion scores) - w_tox * mean(six toxicity scores)
```

Its default weights are `1.0` and `1.2`. It is not an objective assessment of a person, calibrated probability, or accuracy metric. The two replacement models were trained independently, so their score scales are not jointly calibrated. The threshold controls flags; weights affect only this heuristic, not the model predictions.

## Installation and demo

Use Python **3.12** and run all commands from the repository root. The three model weight downloads total approximately 1.5 GB; allow additional disk space for PyTorch and dependencies. CPU inference loads these models into memory. No API key or paid inference service is needed for manual analysis. Optional YouTube retrieval needs your own API key and consumes its API quota.

Windows PowerShell:

```powershell
git clone https://github.com/KALELIN-X/Pytorch-Bert-Pos-Neg-comment-detection.git
cd Pytorch-Bert-Pos-Neg-comment-detection
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.download_models
.\.venv\Scripts\python.exe -m src.download_sentiment
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal at the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run MainFolder/app/Dashboard.py --server.address 127.0.0.1
```

POSIX equivalent after cloning:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m src.download_models
python -m src.download_sentiment
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
# Second terminal, with the same virtual environment activated:
python -m streamlit run MainFolder/app/Dashboard.py --server.address 127.0.0.1
```

Open the dashboard at [localhost:8501](http://127.0.0.1:8501), the API landing page at [localhost:8000](http://127.0.0.1:8000), or interactive API docs at [localhost:8000/docs](http://127.0.0.1:8000/docs). `GET /health` reports readiness, backend, and limits. Missing/corrupt model files leave the API available with `ready: false`; predictions return HTTP 503 with setup guidance. Restart the API after setup. The dashboard shows offline/setup/timeout/validation errors in readable form.

Readiness is independent: `ready` describes the original emotion/toxicity model(s), and `sentiment_ready` describes the newly added sentiment model. One task can work while the other needs setup. The UI displays readiness for the selected task.

Optional configuration: copy `MainFolder/.env.example` to root `.env`. Existing environment variables take precedence. `MODEL_DIR` changes the public model location; `API_URL` changes the dashboard endpoint. Credentials, datasets, weights, logs and interview notes are ignored by Git.

## API examples

For sentence sentiment, use `POST /sentiment` with `{"text":"I love this product. It is wonderful!"}`. It returns `label`, three `scores`, `explanation`, and metadata explicitly identifying uncalibrated scores. `{"texts":[...]}` returns separate ordered results and an `aggregate` object containing `total`, `counts`, and `percentages` for positive/neutral/negative. Sentiment accepts only `text` or `texts`; emotion thresholds and heuristic weights are exclusive to `/predict`.

For example, this batch was exercised with the downloaded pinned model locally:

```json
{"texts":["I love this product. It is wonderful!","This product is terrible and disappointing.","It is a standard product without anything remarkable."]}
```

Observed labels were positive, negative, neutral. The aggregate was three analyzed comments, one of each label, about 33.33% per category. This is a demonstration observation, not an evaluation dataset or accuracy result; neutral or ambiguous text is not guaranteed a particular prediction.

PowerShell single comment:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -Body '{"text":"Thank you for explaining this so clearly. I really appreciate your help!"}'
```

Batch request, preserving input order with a separate result for each comment:

```json
{
  "texts": [
    "Thank you for explaining this so clearly. I really appreciate your help!",
    "You are an idiot. Shut up and stop wasting everyone's time.",
    "Great, another meeting that could have been an email."
  ],
  "threshold": 0.5,
  "w_pos": 1.0,
  "w_tox": 1.2
}
```

The response retains `toxicity`, `positivity`, `toxicity_bin`, `positivity_bin`, and `net_goodness` from the earlier API and adds `label`, `explanation`, and score metadata. A batch wraps these objects in `results`, includes each zero-based `index`, and returns `count`. Full scores contain all six labels in each group. The short excerpt below illustrates the response shape; use the running API to see actual scores:

```json
{
  "label": "positive_signals",
  "explanation": "Positive emotion signals (gratitude) meet the selected threshold.",
  "metadata": {
    "score_type": "uncalibrated_model_scores",
    "confidence_calibrated": false,
    "threshold": 0.5,
    "w_pos": 1.0,
    "w_tox": 1.2
  }
}
```

`examples/comments.json` contains synthetic positive, hostile, and ambiguous demo text. It is not a labeled benchmark. Sarcasm and ambiguity have no guaranteed label.

Validation: exactly one of `text` (string) or `texts` (1–20 strings), nonempty after trimming, at most 2,000 raw characters and 192 tokenizer tokens per comment (including special tokens). A custom checkpoint can impose a smaller token limit. Inputs are rejected rather than silently truncated during prediction. JSON must be an object, unknown fields are rejected, and the body is limited to 64 KiB even without a Content-Length header. Threshold is a finite number in `[0, 1]`; each heuristic weight is finite in `[0, 3]`. Strings and booleans are not accepted as numeric settings. Invalid input returns 422, oversized bodies 413, and model/prediction failures 503. Error responses omit rejected values and internal exception details.

## Post comments: paste, CSV, or optional YouTube

Select **Post comments** in the dashboard. Paste one comment per line (blank separator lines are skipped), or upload a UTF-8 CSV up to 256 KiB containing exactly one `text`, `comment`, or `comment_text` column. CSV quoting preserves multiline comments. Up to 100 comments are analyzed for sentiment, in sequential API batches of at most 20. The original emotion/toxicity task remains limited to 20 comments. Invalid/oversized sets are rejected clearly. Duplicate comments remain separate comments in these summaries. No partial percentage summary is shown when a prediction batch fails.

YouTube retrieval uses [the official commentThreads.list API](https://developers.google.com/youtube/v3/docs/commentThreads/list). Enable YouTube Data API v3 in your own Google project and set `YOUTUBE_API_KEY` in the ignored root `.env`, then restart the API. No key is requested in the dashboard or included in Git. `GET /health` exposes only whether retrieval is configured.

```text
POST /post-comments/youtube
{"url":"https://www.youtube.com/watch?v=VIDEO_ID","max_comments":100}
```

Replace `VIDEO_ID` with an actual 11-character video ID. The endpoint accepts validated HTTPS watch/shorts/embed/youtu.be URLs and makes **one** request to Google's fixed API endpoint: newest top-level comments, at most 100, plain text. It does not scrape the user URL, follow redirects, retrieve replies, or paginate. `has_more`, received/retrieved counts and excluded blank/oversized counts describe the sample. Text over 2,000 characters is excluded without truncation; model-token oversize still produces a clear prediction error. Results describe the analyzed subset, never all comments on the post. Comments-disabled/quota/access/network failures return safe messages and a manual-import fallback. Responses are bounded and keys/upstream exception details are not exposed.

A locally configured key was exercised against Google's official API, which returned HTTP 403 access denied. Successful live retrieval remains unverified and requires a Google project/key permitted to use YouTube Data API v3. Parsing, request bounds and failure cases are covered by offline tests; manual imports and actual local sentiment predictions are verified. Instagram retrieval is not implemented: use an authorized exported/pasted comment set for that platform. No account scraping or bypass is included.

## Repaired shared-BERT training

Training is optional; the default demonstration does not require datasets or fine-tuning. The original shared encoder/two-head architecture is retained. Improvements include:

- Required, validated binary labels and nonblank text; missing task labels remain masked during multi-task BCE loss.
- Seeded normalized-text-group split across sources, keeping identical comments out of both training and validation.
- Class weights calculated from training rows only using negative/positive counts; AdamW, linear warmup/decay, gradient clipping and non-finite loss checks. Each label must have both classes in training; insufficient coverage is rejected.
- Validation micro/macro F1 and per-label precision/recall/F1/support at threshold 0.5. The best mean task macro-F1 selects the checkpoint.
- Portable `model.safetensors`, local encoder config/tokenizer, metadata, dataset hashes, split counts, hyperparameters and training history. No raw comments or credential-bearing data paths are saved in metadata.

Supply your own authorized public dataset files locally. The six-label format requires [Jigsaw's 2018 Toxic Comment Classification data](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data), not the single-label Jigsaw multilingual competition linked in the original notes. [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) must be expanded into binary emotion columns; the trainer rejects label-ID-list CSVs or incompatible mirrors instead of guessing a conversion. Review each source's access and usage terms. This repository does not redistribute datasets.

Required CSV headers (extra columns are ignored):

```text
Jigsaw: text,toxic,severe_toxic,obscene,threat,insult,identity_hate
GoEmotions: text,admiration,gratitude,joy,pride,love,optimism
```

`comment_text` may replace `text`; every required label must be a nonmissing 0 or 1. Do not use unlabelled competition test rows with `-1` labels. A split requires enough distinct comment groups for both sources on both sides. Training truncates examples to `--max-len`; serving rejects oversized comments. The group split is not author/thread-aware or multilabel-stratified, so rare labels may be absent from validation. Zero-denominator metrics use zero and report support. Validation is used for checkpoint selection, so its results are not independent test accuracy.

Windows, after core installation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m src.train_multitask --jigsaw-csv MainFolder/data/jigsaw_train.csv --goemotions-csv MainFolder/data/goemotions.csv --epochs 3 --batch-size 16 --lr 2e-5 --max-len 192 --seed 42 --device cpu
```

CPU fine-tuning can be slow; choose a smaller batch on limited hardware. The default backbone is Apache-2.0 `google-bert/bert-base-uncased`, pinned to `86b5e0934494bd15c9632b12f734a8a67f723594`. Changing the model requires changing its immutable revision too. See `python -m src.train_multitask --help` for all controls. No full-dataset training was performed as part of this reconstruction.

The default output is `MainFolder/artifacts/civicbalance/`. To serve your newly trained checkpoint, set the backend in the API terminal and restart:

```powershell
$env:INFERENCE_BACKEND = 'shared_bert'
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Set `ARTIFACT_DIR` to the parent directory of `civicbalance` for a different location. `INFERENCE_BACKEND=public` restores the demonstration backend. Both return the same API schema. Historical `.pt` files are not loaded: missing original weights are not recovered or represented as newly trained weights.

Optional Postgres exploration remains in `MainFolder/sql/`, `src/etl.py`, and `src/query_examples.py`. Install training extras, configure a local `DATABASE_URL`, initialize `MainFolder/sql/schema.sql`, then run `python -m src.etl` or `python -m src.query_examples`. The original importer is append-only and is not required for training or predictions; avoid repeated imports into an existing populated database. Database integration is optional and was not exercised against a live Postgres instance.

## Tests and verification

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-training.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m src.eval
```

On POSIX use `python` from the activated environment. The tests run offline: API/model tests use deterministic fixtures for workflow, batch ordering, validation, token boundaries, corrupt/nonfinite output and safe failure handling. Training tests exercise actual optimizer updates, masked labels, reproducible splits, train-only weights, metric calculations and local checkpoint roundtrips using a small synthetic BERT fixture. Synthetic fixture metrics demonstrate code behavior; they are not evidence of real-world model accuracy. `src.eval` prints demonstration predictions, not a dataset accuracy benchmark.

Tests also cover true three-class sentiment, count-based post percentages across multiple API batches, withholding totals on partial failures, CSV parsing, platform URL/request bounds and safe YouTube failures. `pytest.ini` uses the ignored `.pytest-tmp` folder for test artifacts, avoiding dependence on the system temporary directory. `examples/sentiment_comments.csv` is a synthetic import/demo file, not a labeled benchmark.

The API and dashboard are also verified locally using downloaded pinned models. No labeled evaluation dataset is checked in, so **no application accuracy figure is claimed**. The original `MainFolder/reports` files remain explicitly identified as blank historical templates.

## Model provenance and limitations

The exact model IDs, immutable revisions, label mappings, source descriptions and licenses are recorded in `src/model_manifest.json`. The downloader fetches only configuration/tokenizer/safetensors files anonymously; it does not execute remote model code. A local download manifest verifies the expected IDs and revisions before loading.

The separate sentiment model is recorded in `src/sentiment_manifest.json` and downloaded with `python -m src.download_sentiment`. It is [lxyuan/distilbert-base-multilingual-cased-sentiments-student](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student), Apache-2.0, pinned to `cf991100d706c13c0a080c097134c05b7f436c45`. Its model card describes distilling a zero-shot multilingual teacher on the multilingual-sentiments corpus, rather than training directly on its human annotations. The model-card teacher/student agreement is not labeled accuracy and is not a measurement of this application. This new classifier supports positive/neutral/negative labels but is not part of the original shared-BERT trainer. Real smoke checks showed that factual sentences can skew positive; neutral handling, context and sarcasm remain limitations.

| Demonstration model | Revision | License / source |
| --- | --- | --- |
| [unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert) | `4d6c22e74ba2fdd26bc4f7238f50766b045a0d94` | Apache-2.0; model card identifies Jigsaw toxicity data |
| [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions) | `d75048347613a25d77de8cf6412eaae9fa7b26be` | MIT; English Reddit GoEmotions |

The Hugging Face toxicity variant differs from Detoxify releases; published Detoxify or model-card results are not measurements of this reconstruction. English Wikipedia/Reddit training domains may differ from your comments. Context, sarcasm, quoting, dialect, identity references and domain shifts can cause errors. The score-based explanation reports labels crossing the threshold; it is not token attribution or a causal explanation. The models and weighted score have not been calibrated for moderation decisions. Reasonable next steps are an authorized held-out benchmark, careful error analysis, per-label threshold calibration, and independent test evaluation after training.
