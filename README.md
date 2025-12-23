# Ashwam Journaling - Language Detection System

A deterministic, explainable, and robust language + script detection system designed for short, messy journaling text.

## Approach & Philosophy

Journaling entries are often short, informal, and code-switched (mixed languages). Traditional ML models can be overkill or unpredictable for such "messy" data.

Our approach uses **Signal-Based Heuristics**:
1.  **Detective Work, Not Black Box**: We gather multiple weak signals (script ratios, unique character counts, lexicon hits) and combine them.
2.  **Determinism**: The same input always yields the same result. No random seeds or model drift.
3.  **Explainability**: Every decision produces an `evidence` object, showing exactly *why* a decision was made (e.g., "70% Hindi lexicon match").

## How It Works

### 1. Hinglish vs. English
Distinguishing Romanized Hindi (Hinglish) from English is the hardest challenge. We use a **Dominance & Ratio** strategy:

*   **Lexicon Matching**: We check tokens against a curated list of common Hindi words (e.g., *hai, mein, kyun*) and English stopwords (e.g., *the, is, and*).
*   **The "3x Rule"**: To decide, we look for **dominance**:
    *   If **English Ratio > 3x Hindi Ratio**, it's **English** (ignoring minor overlaps like "to").
    *   If **Hindi Hits > 3x English Hits**, it's **Hinglish**.
*   **Mixed Language**: If *neither* dominates and both have strong signals (> 15-30%), we classify as **Mixed**.

### 2. Confidence Computation
Confidence is not a measure of probability, but of **signal strength**. It ranges from 0.0 to 0.95.

*   **Base Score**: Starts at 0.3.
*   **Boosts**:
    *   Longer text (+0.1 to +0.2).
    *   Strong script dominance (+0.1).
    *   High lexicon matching ratio (+0.15 to +0.3).
*   **Penalties**:
    *   Very short text (< 6 tokens): **-0.05 to -0.15** penalty.
    *   Mixed language: **-0.1** penalty (inherently ambiguous).
    *   Noise (emoji/numbers): **-0.3 to -0.4** penalty.
*   **Caps**:
    *   **0.95**: Global maximum (never 100% sure).
    *   **0.85**: Maximum for short Hinglish or Mixed text.

## Usage

### Prerequisites
*   Python 3.7+
*   No external dependencies (Standard Library only)

### Running Detection
Use the CLI tool `lang_detect.py`:

```bash
# Process a file
python3 lang_detect.py --in texts.jsonl --out results.jsonl

# See statistics (optional)
python3 lang_detect.py --in texts.jsonl --out results.jsonl --stats
```

**Input Format (`.jsonl`)**:
```json
{"id": "1", "text": "aaj mausam accha hai"}
```

**Output Format**:
```json
{
  "id": "1", 
  "primary_language": "hinglish", 
  "script": "latin", 
  "confidence": 0.90, 
  "evidence": {
    "hi_lexicon_ratio": 0.75, 
    "n_tokens": 4, 
    ...
  }
}
```

### Running Tests
We have a comprehensive test suite covering edge cases, regression, and logic verification.

```bash
# Run all tests
python3 test_detector.py
```

## Limitations & Known Failure Cases

1.  **Short Ambiguous Text**: Inputs like "hi" or "to" exist in both languages. We default to English or Unknown with very low confidence.
2.  **Lexicon Dependency**: If a Hinglish sentence uses valid Hindi words *not* in our small lexicon (e.g., obscure nouns), it may be misclassified as English or Unknown.
3.  **Spelling Variations**: "kya" vs "kyaaa" - strict matching might miss informal spelling variations (though we cover common ones).
4.  **Code-Switching Complexity**: Sentences that switch languages mid-phrase are hard to boundary-check without a sequence model.

## Future Improvements

If we had more time or tools, we would:

1.  **Phonetic Matching**: Use a Soundex-like algorithm for Indian languages to handle spelling variations (e.g., *kaise* vs *kese*).
2.  **N-gram Analysis**: Look at character tri-grams to detect language "texture" beyond word lookups.
3.  **Contextual ML**: Train a small FastText or Naive Bayes model on a larger corpus to handle vocabulary we missed.
4.  **Browser/Edge Support**: Port this logic to TypeScript/WASM for client-side detection to save server costs.
