# Test Scripts

These scripts test the taxonomy generation and SFT dataset tools.

## Prerequisites

1. Copy `env.example` to `.env` and fill in your API keys:
   ```bash
   cp env.example .env
   # Edit .env and set LLM_PROVIDER, LLM_MODEL, and LLM_API_KEY
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Test Scripts

### 1. Test Taxonomy Generation

```bash
python scripts/test_generate_taxonomy.py
```

**What it does:**
- Generates categories for "Kazakhstan"
- Processes only the first 2 categories (early stopping)
- Generates subcategories for those categories
- Generates keywords for all subcategories
- Saves results to `test_outputs/taxonomy_<timestamp>.json`
- Prints comprehensive logs

**Configuration:**
- Edit `TEST_COUNTRY` in the script to test a different country
- Edit `EARLY_STOP_CATEGORIES` to process more/fewer categories

### 2. Test SFT Dataset (Images)

```bash
python scripts/test_sft_images.py
```

**What it does:**
- Loads images from `test_data/images/`
- Annotates them using the LLM
- Generates SFT training examples
- Saves annotations and SFT examples to `test_outputs/`

**Setup:**
1. Add some test images to `test_data/images/` (`.jpg`, `.png`, `.webp`)
2. Optionally organize them in subfolders (folder names become `topic_hint`)

### 3. Test SFT Dataset (Text)

```bash
python scripts/test_sft_text.py
```

**What it does:**
- Loads text files from `test_data/texts/`
- Performs knowledge distillation (LLM reads source, generates standalone Q&A)
- Generates SFT training examples (source text NOT included)
- Saves annotations and SFT examples to `test_outputs/`

**Setup:**
- Sample text files are already in `test_data/texts/`
- Add your own `.txt` files to test with different content

## Output Files

All outputs are saved to `test_outputs/` with timestamps:

- `taxonomy_YYYYMMDD_HHMMSS.json` — full taxonomy results
- `sft_images_annotations_YYYYMMDD_HHMMSS.jsonl` — image annotations
- `sft_images_YYYYMMDD_HHMMSS.jsonl` — image SFT examples
- `sft_text_annotations_YYYYMMDD_HHMMSS.jsonl` — text annotations
- `sft_text_YYYYMMDD_HHMMSS.jsonl` — text SFT examples

## Troubleshooting

**"LLM_API_KEY environment variable is not set"**
- Make sure you created `.env` from `env.example`
- Make sure `.env` is in the project root
- Check that `python-dotenv` is installed (it's in requirements.txt)

**"No images found"**
- Make sure `test_data/images/` exists and contains image files
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

**"No text files found"**
- Make sure `test_data/texts/` exists and contains `.txt` files
