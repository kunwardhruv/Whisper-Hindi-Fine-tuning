# Downloading & Generating Large Files

Some files are too large for GitHub (~2.5GB total). Follow these instructions to get them locally.

## 📦 Large Files Overview

| File | Size | Download/Generate? |
|------|------|-------------------|
| `models/whisper-quick-trained/` | 1.4GB | Generate via training |
| `outputs/spelling_classification.csv` | 25MB | Generate via script |
| `data/audio/` | 800MB+ | Download from Josh Talks |
| `data/transcripts_raw/` | ~50MB | Download from Josh Talks |

---

## 1️⃣ Fine-tuned Model

### Option A: Train Yourself (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run training script
cd notebook
python quick_train.py

# Model will be saved to: models/whisper-quick-trained/
```

**Time**: ~15 minutes | **GPU**: Recommended (CPU takes 1-2 hours)

### Option B: Download Pre-trained (If Available)
If you have the model weights stored on Google Drive/Hugging Face:
```bash
# Example: Download from Hugging Face Hub
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('kunwardhruv/whisper-small-hindi')
model.save_pretrained('models/whisper-quick-trained/')
"
```

---

## 2️⃣ Spelling Classification CSV (25MB)

### Generate Locally
```bash
cd notebook
python question3_spell_check.py

# Output: ../outputs/spelling_classification.csv
# Time: ~5-10 minutes
```

This will create the classification file with 177,508 words.

---

## 3️⃣ Training Data

### From Josh Talks
The training data should be provided by Josh Talks. Expected structure:
```
data/
├── audio/              # 104 WAV files (16kHz)
├── transcripts_raw/    # 104 JSON transcriptions
├── transcripts_txt/    # Converted text files
└── metadata/           # Speaker metadata
```

**Contact**: Ask Josh Talks for the dataset download link if not already provided.

---

## 4️⃣ Other Large Output Files

### Already in GitHub ✅
These files ARE included and ready to review:
- `outputs/eval_all_results.tsv` (2KB) - WER results
- `outputs/error_sample_25.tsv` (15KB) - Error analysis
- `outputs/asr_cleanup_results.json` (50KB) - Cleanup results
- `outputs/question4_lattice_results.json` (30KB) - Lattice WER results
- `outputs/low_confidence_sample.csv` (5KB) - Spell check review
- `outputs/training_results.json` (5KB) - Training metrics
- `outputs/model_card.md` - Model documentation
- `outputs/evaluation_report.md` - Full evaluation report

---

## ✅ Complete Setup in 3 Steps

### Step 1: Clone Repository
```bash
git clone https://github.com/kunwardhruv/Josh-talks-assignment-.git
cd Josh-talks-assignment-
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Large Files (Optional)
```bash
# Train model
cd notebook && python quick_train.py

# Generate spelling classification
python question3_spell_check.py

# Evaluate
python final_evaluation.py
```

---

## 🔍 What Josh Can Review Right Now

✅ **No downloads needed** to see these on GitHub:
1. **ASSIGNMENT_SUBMISSION.md** - Complete Q1-Q4 answers
2. **README.md** - Quick start guide
3. **All Python code** - Implementation details
4. **Output CSVs/JSONs** - Results & metrics
5. **Model card** - Model documentation
6. **.gitignore** - This file

⬇️ **Josh needs to generate locally** (or request):
1. Fine-tuned model weights (1.4GB)
2. Full spelling classification CSV (25MB)
3. Training dataset (800MB+)

---

## 📝 Summary for Josh Talks Evaluator

**What's on GitHub**: Complete assignment answers + all results files  
**What's not**: Large models & full dataset (reproduce locally or request)  
**Time to setup**: 5 minutes (clone) + 15 minutes (train) = 20 minutes  
**Recommendation**: Start with GitHub review, generate large files if needed for verification

---

## 🆘 Troubleshooting

### Model file not found?
```bash
# Check if model exists
cd models
ls -la whisper-quick-trained/
```
If missing, run `python notebook/quick_train.py` to regenerate.

### spelling_classification.csv missing?
```bash
# Regenerate
python notebook/question3_spell_check.py
```

### Data files missing?
Contact Josh Talks for download link or mount from shared drive.

---

**Questions?** See README.md or run `python -m ipynb notebook/download_data.ipynb` to download data.
