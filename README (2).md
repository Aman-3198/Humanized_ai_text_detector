# Humanized AI Text Detector

A web tool that reads a piece of text and estimates whether it is **plain AI**, **humanized AI** (AI text reworded to avoid detection) or **human-written**. It has three parts: a trained scikit-learn model, a FastAPI backend that serves it, and a single-page frontend.

> **Status: research prototype.** The model works well on the kinds of text it was trained on and poorly on humanizers it has not seen. See [Known limitations](#known-limitations) before relying on any result.

## Project structure

| File | Purpose |
|---|---|
| `index.html` | Frontend. Sends the text to the API and draws the gauges, verdict and sentence strip. |
| `main.py` | FastAPI backend. Loads the model once and exposes `/api/analyze` and `/health`. |
| `detector_v5.py` | Model class (cleaning, windowing, scoring). |
| `ling.py` | Grammar and word-frequency features (spaCy + wordfreq). |
| `detector_v5.joblib` | Trained model. |
| `requirements.txt` | Python dependencies. |

## Quick start

```
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Open `http://localhost:8000`. Opening `index.html` directly from disk also works; it then calls `http://localhost:8000`. scikit-learn is pinned to 1.8.0 because that is the version the saved model was trained with.

## API

`POST /api/analyze` with `{"text": "..."}` (at least 8 words, at most 20,000 characters).

```json
{
  "ai": 1, "human": 97, "hum": 2,
  "top": "human", "label": "human", "uncertain": false, "confidence": 0.964,
  "model": "detector_v5",
  "sentences": [{"text": "...", "words": 21, "ai": 2, "human": 96, "hum": 2, "top": "human", "confidence": 0.96}]
}
```

`ai`, `human` and `hum` (humanized AI) are whole percentages that sum to 100. `label` is `uncertain` when the top class is below 55%. `GET /health` reports whether the model is loaded.

## Datasets

| Dataset | Size | Contents |
|---|---|---|
| `updated_dataset_final_8079.csv` | 8,079 rows | 3,080 AI texts and 4,999 human texts. Short answers (about 320 characters for AI) to Reddit-style questions. AI writers: GPT (1,000), Grok (990), Claude (79), Gemini (11), unlabelled (1,000). |
| `text_detection_15000.csv` | 14,999 usable rows | 5,000 AI, 5,000 AI-humanized, 4,999 human. Long passages (about 1,500 characters). AI writers: GPT-2, MPT, MPT-chat, Llama-chat, Mistral, Mistral-chat. Humanized texts are the AI texts with words swapped for synonyms. Four malformed rows were dropped. |
| `humanized_detection_dataset.csv` | 4,238 rows | Built from the 8,079 file: 1,780 AI texts that have at least one humanized version, plus their 2,458 humanized versions. The 1,300 AI texts with no humanization were removed. |
| `humanized_pairs.csv` | 2,458 rows | One row per humanized text, with its original AI text beside it. |
| `combined_clean_dataset.csv` | 15,994 rows | The training set for v5: both datasets after cleaning and de-duplication (table below). |
| 20 base passages and 5 closers (docx) | 100 composed texts | Rewrites of the 20 base passages and 5 closing lines behind IDs 6327 to 6989, combined into 100 texts. Seven more come from `humanized_batch_3_data.txt`. |

Composition of `combined_clean_dataset.csv`:

| Source | Plain AI | Humanized AI | Human |
|---|---|---|---|
| Reddit-style set (8,079 file) | 1,481 | 1,795 | 4,936 |
| Long-passage set (15,000 file, half of the topics) | 2,646 | 2,652 | 2,484 |

## Humanizers

The author ran the AI texts through the humanizer tools found on the first two pages of Google results for AI humanizers. The data files do not record which tool produced which output, so the tool names are not listed here yet. The synonym-swap humanizer in the 15,000-row file came with that dataset and is not one of these tools.

**Tool names:** _to be added by the author_

| Output | Original IDs | Humanized texts | Used for |
|---|---|---|---|
| `humanized.txt` | 6327 to 6989 | 663 | Training |
| `humanized_batch_3` markdown file | 6327 to 6989 | 663 | Training |
| "Untitled document" docx | 5001 to 5663 | 587 | Training |
| `humanize_2.pdf` | 6990 to 7652 | 545 | Training |
| Synonym-swap humanizer (inside the 15,000 file) | not applicable | 5,000 | Training |
| 20 base passages and 5 closers docx, plus `humanized_batch_3_data.txt` | 6327 to 6989 | 107 | Unseen test first, then added to the final training |

Left out: `grok_report.pdf` (an empty page with only a download card), about 180 humanized entries with no ID, and five entries whose text did not match their ID (5750, 7097, 7238, 7258, 7532).

## Data preparation

1. **Cleaning.** Invisible zero-width characters are removed, curly quotes and dashes are made plain, and whitespace is collapsed.
2. **Detokenizing.** 88% of the human answers in the 8,079 file had spaces before punctuation ("weight , not"), against 8% of the AI texts. A model could learn that spacing instead of real writing differences, so all text is detokenized before training and at prediction time.
3. **Windows.** Texts are cut into windows of about 300 characters, at most four per text, so short answers and long passages look alike to the model. At prediction time the window scores are averaged.
4. **Group-aware splits.** Texts that share an original passage or topic never appear on both sides of a split.

## Model

Each window is scored by three logistic regressions whose log-probabilities are averaged:

- word TF-IDF (1 to 2 grams)
- character TF-IDF (3 to 5 grams)
- grammar and word-rarity features: part-of-speech and dependency rates from spaCy, and word frequency from the wordfreq library

Classes are weighted to balance them. A text whose top class is under 55% is reported as uncertain.

## Evaluation

All numbers are on held-out data. Accuracy is at text level.

| Test | Result |
|---|---|
| New topics and passages, all humanizers seen | 90.0% accuracy, AUC 0.98 (plain AI vs humanized). Recall: AI 0.88, humanized 0.87, human 0.93. |
| Unseen humanizer: docx set | AUC 0.81, humanized recall 0.26, human recall 0.99 |
| Unseen humanizer: PDF set | AUC 0.99, humanized recall 0.21, human recall 0.99 |
| Unseen humanizer: txt and markdown sets | AUC 0.83, humanized recall 0.48, human recall 0.99 |
| Unseen humanizer: synonym swap | AUC about 0.60 |
| Unseen humanizer: 20 passages docx | AUC 0.59. 78 of 100 rewrites were called human. |
| Hand test: a personal text known to be humanized | Called human at 97% (wrong) |

Rows for the PDF and for txt and markdown use the word and character models only. The docx row also holds with the grammar features added.

## Known limitations

- **Unseen humanizers are the main weakness.** Each humanizer leaves its own marks and the model learns the ones it has seen. Casual, conversational rewriting makes text look human to it.
- **Personal and opinion writing is under-represented** in the training data, and this is where it fails most.
- **The plain-AI examples are narrow.** The Reddit-style set has about 800 distinct AI texts, and IDs 6327 to 6989 are built from just 20 base passages.
- **Confidence is not accuracy.** A score of 97% human was wrong on a known humanized text.
- **Sentence colours are rough.** Single sentences are much shorter than the 300-character windows the model was trained on.

The most useful next step is more data: several hundred humanized personal, opinion and factual texts, from many different tools, each with its human or AI original.

## Retraining

`combined_clean_dataset.csv` has the columns `text_clean`, `label`, `dataset`, `source` and `group`. Split on `group`. Build the 300-character windows with `chunks()` and the cleaning with `clean()` from `detector_v5.py`, compute the features with `ling_feats()` from `ling.py`, then call `DetectorV5().fit(windows, features, labels)`.
