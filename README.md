# (ISBI 2026 - inference code)

This repository provides the official deployment and inference codebase for our **ISBI 2026** submission:  
**"Time-Aware Multimodal Fusion of Mammograms and Clinical Notes for Breast Cancer Patient Retrieval"**.

---

## 📌 Architecture Overview

The framework performs multimodal patient retrieval by aligning multi-view mammography exams with longitudinal clinical notes under time-discrepancy awareness:

```
                  ┌───────────────────────────────┐
                  │ 4 Mammogram Views             │
                  │ (CC_L, MLO_L, CC_R, MLO_R)    │
                  └───────────────┬───────────────┘
                                  │ EfficientNet-B5
                                  ▼
                  ┌───────────────────────────────┐
                  │ View Pooling               │──► Unimodal Visual Embedding (2048-d)
                  └───────────────┬───────────────┘
                                  │ visual_proj (1024-d)
                                  ▼
┌─────────────────────────┐   ┌───────────────────────────────┐
│ Time Discrepancy Gate   ├──►│ Bidirectional Cross Attention │──► Multimodal Embedding (1024-d)
│ g = max(0, 1 - Δt / 2)  │   │ (BCA) + LayerNorm             │
└─────────────────────────┘   └───────────────▲───────────────┘
                                              │ text_proj (1024-d)
                  ┌───────────────────────────┴───┐
                  │ TemporalTextFusion            │──► Unimodal Text Embedding (768-d)
                  └───────────────▲───────────────┘
                                  │ ModernBERT-base
                  ┌───────────────┴───────────────┐
                  │ Longitudinal Clinical Notes   │
                  │ (Newline-segmented visits)    │
                  └───────────────────────────────┘
```

- **Image Backbone**: EfficientNet-B5 extracted from Mammo-FM (2048-dim output).
- **Text Backbone**: ModernBERT-base (768-dim per visit segment).
- **Temporal Text Fusion**: TransformerEncoder with learnable `[CLS]` token and time-delta positional projection.
- **Multimodal Fusion**: Bidirectional Cross Attention (BCA) modulated by a continuous time-decay gate $g = \max(0, 1 - \frac{\Delta t}{2})$ (where $\Delta t$ is the discrepancy in years between mammogram acquisition and diagnosis date).
- **Standalone Instantiation**: The model architecture is built directly without requiring any multi-gigabyte base pretraining archive (`Mammo-FM_ASU_Trained_CLIP.tar`). All weights are directly hydrated from the unified multimodal checkpoint.

---

## 📂 Repository Structure

```text
codebase/
├── README.md                      # This documentation
├── inference_emory.py             # Easy-to-run inference & embedding extraction script
├── environment_inference.yml      # Conda environment specification
├── breastclip/                    # Model modules (EfficientNet & transformer definitions)
├── checkpoints/                   # Checkpoints directory (place downloaded weights here)
│   └── time_aware_multimodal_fusion.pth
├── data/                          # Input data directory
│   └── test_cohort.csv            # Input cohort metadata & note summaries
└── data_csv/                      # Existing evaluation datasets
```

---

## ⚙️ 1. Environment Installation

### Option A: Conda (Recommended)

Create and activate the environment using [`environment_inference.yml`](environment_inference.yml):

```bash
# 1. Create environment
conda env create -f environment_inference.yml

# 2. Activate environment
conda activate mammo_infer
```

### Option B: Python Virtual Environment (`venv` + `pip`)

If you prefer `venv`:

```bash
python3.11 -m venv mammo_infer_env
source mammo_infer_env/bin/activate

# Install PyTorch with CUDA 12.1+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install required dependencies
pip install "transformers>=4.45.0" "timm>=1.0.0" opencv-python-headless "pydicom>=2.4.0" "dicomsdl>=0.109.0" pillow pandas numpy tqdm
```

### Verification

Check that CUDA and the core packages load cleanly:

```bash
python -c "import torch, transformers, timm, cv2, pydicom, dicomsdl; print('Environment ready! CUDA:', torch.cuda.is_available())"
```

---

## 📥 2. Download Model Checkpoint

The trained ISBI 2026 multimodal checkpoint is shared via Google Drive:

* **Checkpoint Download Link**: [Google Drive Link](https://drive.google.com/file/d/1pPiK1YZ7U_52vrrw71HuSz8D0MaK1bbf/view?usp=drive_link)
* **File Name**: `Emory_BCA_True_Happier_fp16_frozen_timedecay_flipr2l_epoch1_LRdecay_valloss0.4883.pth`

### Directory Placement

Create the `checkpoints/` directory and place the checkpoint inside it:

```bash
mkdir -p checkpoints

# Option 1: Using gdown directly in terminal
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/1pPiK1YZ7U_52vrrw71HuSz8D0MaK1bbf/view?usp=drive_link" -O checkpoints/time_aware_multimodal_fusion.pth

# Option 2: If downloaded manually in your browser, move and rename it:
# mv ~/Downloads/Emory_BCA_True_Happier_MeanPooling_*.pth checkpoints/time_aware_multimodal_fusion.pth
```

> **Note**: The script automatically checks `checkpoints/time_aware_multimodal_fusion.pth` by default, but you can also pass any custom checkpoint path using `--ckpt <path_to_checkpoint.pth>`.

---

## 📊 3. Input CSV Format (Minimal Expected Input)

The inference script accepts a standard `.csv` file. The table below outlines the minimal expected columns:

| Column Name | Type | Requirement | Description |
| :--- | :--- | :--- | :--- |
| `PATIENT_CLINIC_NUMBER` | `str` / `int` | **Required** | Unique patient or exam identifier. *(Aliases accepted: `patient_id`, `PatientID`)*. |
| `CC_L` | `str` | **Required** | File path to Cranio-Caudal Left image (`.dcm`, `.png`, or `.jpg`). |
| `MLO_L` | `str` | **Required** | File path to Medio-Lateral Oblique Left image (`.dcm`, `.png`, or `.jpg`). |
| `CC_R` | `str` | **Required** | File path to Cranio-Caudal Right image (`.dcm`, `.png`, or `.jpg`). |
| `MLO_R` | `str` | **Required** | File path to Medio-Lateral Oblique Right image (`.dcm`, `.png`, or `.jpg`). |
| `summary` | `str` | **Required** | Timeline of clinical history/notes. Sequential visits should be separated by newline (`\n`). Date headers such as `Date(0) (initial diagnosis): ...` or `Date(30) after (followup): ...` are parsed for temporal delta encoding; plain lines default to delta `0.0`. |
| `Time_discrepancy` | `float` | **Recommended** | Time difference (in years) between the mammogram study date and the diagnosis/reference date. Used for dynamic time-decay cross-attention gating. Defaults to `0.0` if omitted or NaN. |

*Robust Fallbacks*: Missing image files or `NaN` paths are automatically caught and substituted with zero-tensors, ensuring the batch extraction never halts.

### Sample CSV Row Example

```csv
PATIENT_CLINIC_NUMBER,CC_L,MLO_L,CC_R,MLO_R,summary,Time_discrepancy
1001,/path/to/1001_CC_L.dcm,/path/to/1001_MLO_L.dcm,/path/to/1001_CC_R.dcm,/path/to/1001_MLO_R.dcm,"Date(0) (initial diagnosis): Right breast mass palpated.\nDate(28) after (biopsy): Invasive ductal carcinoma confirmed.",0.15
```

---

## 🚀 4. Running Inference & Embedding Extraction

### Command-Line Interface (CLI)

Run inference directly on your test cohort:

```bash
python inference_emory.py \
    --csv data/test_cohort.csv \
    --ckpt checkpoints/time_aware_multimodal_fusion.pth \
    --direction r2l \
    --batch_size 4 \
    --gate dynamic \
    --output extracted_embeddings.npz
```

#### CLI Options

- `--csv`: Path to the input CSV file *(default: automatically detects `data/test_cohort.csv` or existing test dataset in `data_csv/`)*.
- `--ckpt`: Path to the multimodal `.pth` checkpoint *(default: `checkpoints/time_aware_multimodal_fusion.pth`)*.
- `--direction`: Laterality flipping alignment (`r2l`, `l2r`, or `none`). Default is `'r2l'` (matching the training alignment).
- `--batch_size`: Batch size for DataLoader *(default: `4`)*.
- `--gate`: Time-gating mode: `'dynamic'` (time discrepancy decay) or `'static'` (forces `gate=1.0`).
- `--output`: File path to save embeddings (`.npz` or `.pt`). Default is `extracted_embeddings.npz`.
- `--num_workers`: Number of background DataLoader workers *(default: `2`)*.

---

### Python API

You can also run inference programmatically within Python:

```python
from inference_emory import evaluate_model

# 1. Extract embeddings
results = evaluate_model(
    test_csv_path="data/test_cohort.csv",
    ckpt_path="checkpoints/time_aware_multimodal_fusion.pth",
    direction="r2l",             # Laterality flipping
    batch_size=4,
    gate="dynamic",              # Time-aware decay gating
    output_path="extracted_embeddings.npz"
)

# 2. Access extracted arrays
patient_ids = results["patient_ids"]              # List[str] of shape (N,)
multimodal_emb = results["multimodal_embeddings"] # np.ndarray of shape (N, 1024) [L2-normalized]
visual_emb = results["visual_embeddings"]         # np.ndarray of shape (N, 2048) [L2-normalized]
text_emb = results["text_embeddings"]             # np.ndarray of shape (N, 768)  [L2-normalized]

print(f"Extracted {len(patient_ids)} patients.")
print(f"Multimodal Embedding Shape: {multimodal_emb.shape}")
```

---

## 🔍 5. Patient Retrieval Evaluation

Once embeddings are extracted, you can evaluate patient retrieval (e.g., patient-to-patient cosine similarity and Recall@$K$):

```python
import numpy as np

# Load extracted embeddings
data = np.load("extracted_embeddings.npz")
patient_ids = data["patient_ids"]
multimodal = data["multimodal_embeddings"]  # (N, 1024), already L2-normalized

# Compute pairwise cosine similarity matrix
similarity_matrix = multimodal @ multimodal.T  # (N, N)

# Retrieve Top-5 most similar patients for each patient query (excluding self)
top_k = 5
retrieved_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:top_k + 1]

for i, pid in enumerate(patient_ids[:3]):
    top_neighbors = [patient_ids[idx] for idx in retrieved_indices[i]]
    print(f"Query Patient {pid} -> Top-{top_k} Retrieved: {top_neighbors}")
```



