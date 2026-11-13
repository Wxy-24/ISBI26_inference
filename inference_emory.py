"""
==============================================================================
Inference Script for Multimodal Mammography + Clinical Notes Fusion Model
==============================================================================
Architecture:
  - Image Encoder: EfficientNet-B5 (Mammo-FM backbone, 2048-dim output)
  - View Fusion: ViewMeanPooling (parameter-free average pooling across 4 views)
  - Text Encoder: ModernBERT-base (768-dim output per visit segment)
  - Temporal Text Fusion: TransformerEncoder with learnable CLS and time-projection
  - Multimodal Fusion: Bidirectional Cross Attention (BCA) with time-decay gating
    (1024-dim fused multimodal embedding)

==============================================================================
MINIMAL EXPECTED INPUT FROM CSV FILE:
==============================================================================
The CSV file passed to `evaluate_model()` or specified via `--csv` should contain
the following columns per row / patient:

1. 'PATIENT_CLINIC_NUMBER' (str or int) [REQUIRED / PRIMARY KEY]:
   Unique identifier for the patient or exam record.
   Fallback columns accepted if missing: 'patient_id', 'PatientID', or row index.

2. Mammogram View Image Paths [REQUIRED - 4 views]:
   - 'CC_L'  : Path to Cranio-Caudal Left image (.dcm, .png, or .jpg)
   - 'MLO_L' : Path to Medio-Lateral Oblique Left image (.dcm, .png, or .jpg)
   - 'CC_R'  : Path to Cranio-Caudal Right image (.dcm, .png, or .jpg)
   - 'MLO_R' : Path to Medio-Lateral Oblique Right image (.dcm, .png, or .jpg)
   * Note: Missing file paths or NaN values are gracefully handled with a
     zero-tensor fallback so extraction does not crash.

3. 'summary' (str) [REQUIRED]:
   Visit-segmented clinical history/notes.
   Sampling window: Clinical notes are sampled from 3 months before diagnosis (-3m)
   to 2 months after diagnosis (+2m) relative to DX_DATE.
   Standard format has newline ('\\n') separated visits with optional date headers:
     "Date(0) (initial diagnosis): <clinical note for visit 1>
      Date(30) after (followup date): <clinical note for visit 2>
      Date(180) after (chemotherapy note): <clinical note for visit 3>"
   * Note: If 'Date(...)' tags are absent, the script treats each non-empty
     newline-separated line as a sequential clinical visit with time_delta=0.

4. 'Time_discrepancy' (float or int) [RECOMMENDED / OPTIONAL]:
   Time discrepancy (in years) between mammogram acquisition and the clinical
   diagnosis/reference date. No specific date window is required for mammograms
   because this time-aware discrepancy dynamically modulates the fusion gate:
     - time_discrepancy > 1.0  -> gate = 0.0 (image contribution zeroed out)
     - time_discrepancy <= 1.0 -> gate = 1.0 - time_discrepancy / 2.0 (linear decay)
   * Note: If omitted or NaN, defaults to 0.0 (full contribution, gate = 1.0).

Any additional clinical/label columns (such as 'Stage_Label', 'Recurrence', etc.)
are optional and ignored during inference.
==============================================================================
"""

import os
import sys
import contextlib
import warnings
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
cv2.setNumThreads(0)
import numpy as np
import pandas as pd
import pydicom
import dicomsdl
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedModel

# Ensure codebase root is in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

try:
    from breastclip.model import build_model
except ImportError:
    build_model = None
from breastclip.model.modules import EfficientNet


# ==============================================================================
# DEFAULT PATHS & DEVICE SETUP
# ==============================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_MULTIMODAL_CKPT_CANDIDATES = [
    "checkpoint/adapt_to_your_local_environment/"
    "Emory_BCA_True_Happier_fp16_frozen_timedecay_flipr2l_epoch1_LRdecay_valloss0.4883.pth.pth",
]

DEFAULT_CSV_CANDIDATES = [
    "data/test_cohort_example.csv",
]


def resolve_file_path(path: str) -> str:
    """Resolve file path by checking absolute path, CWD, script dir, and repo root."""
    if not path:
        return path
    if os.path.exists(path):
        return os.path.abspath(path)

    script_relative = os.path.join(str(_CURRENT_DIR), path)
    if os.path.exists(script_relative):
        return os.path.abspath(script_relative)

    for base in [
        "/home/xiaoyang/Datacenter_storage/xiaoyang/code/Mammo-FM/src/codebase",
        "/media/Datacenter_storage/xiaoyang/code/Mammo-FM/src/codebase",
    ]:
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    return path


def resolve_default_checkpoint() -> str:
    """Find the default multimodal checkpoint from canonical candidate paths."""
    for candidate in DEFAULT_MULTIMODAL_CKPT_CANDIDATES:
        resolved = resolve_file_path(candidate)
        if os.path.exists(resolved):
            return resolved
    return DEFAULT_MULTIMODAL_CKPT_CANDIDATES[0]


def resolve_default_csv() -> str:
    """Find the default test CSV from canonical candidate paths."""
    for candidate in DEFAULT_CSV_CANDIDATES:
        resolved = resolve_file_path(candidate)
        if os.path.exists(resolved):
            return resolved
    return DEFAULT_CSV_CANDIDATES[0]


DEFAULT_MULTIMODAL_CKPT = resolve_default_checkpoint()


def resolve_base_checkpoint() -> str:
    """Find the Mammo-FM base checkpoint."""
    for candidate in DEFAULT_BASE_CKPT_CANDIDATES:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return DEFAULT_BASE_CKPT_CANDIDATES[0]


# ==============================================================================
# IMAGE PREPROCESSING & LATERALITY FLIPPING
# ==============================================================================
_FLIP_R2L_ALIASES = {"r2l", "right-to-left", "righttoleft", "rtl", "align-left", "left"}
_FLIP_L2R_ALIASES = {"l2r", "left-to-right", "lefttoright", "ltr", "align-right", "right"}
_FLIP_NONE_ALIASES = {"none", "off", "no", "false", "0", ""}


def normalize_flip_direction(direction: Optional[Union[str, bool]]) -> Optional[str]:
    """Canonicalize flip direction argument to 'r2l', 'l2r', or None."""
    if direction is None or direction is False:
        return None
    if direction is True:
        return "r2l"
    key = str(direction).strip().lower().replace("_", "-").replace(" ", "")
    if key in _FLIP_NONE_ALIASES:
        return None
    if key in _FLIP_R2L_ALIASES:
        return "r2l"
    if key in _FLIP_L2R_ALIASES:
        return "l2r"
    raise ValueError(
        f"Unknown flip direction: {direction!r}. Expected one of: 'r2l', 'l2r', 'none' (or None)."
    )


def _laterality_should_flip(img: np.ndarray, direction: str) -> bool:
    """Determine whether image laterality requires horizontal flipping."""
    if img.ndim == 2:
        _, w = img.shape
        img_r = img[:, int(w * 0.25):]
        img_l = img[:, :int(w * 0.75)]
    elif img.ndim == 3 and img.shape[0] in (1, 3):
        _, _, w = img.shape
        img_r = img[:, :, int(w * 0.25):]
        img_l = img[:, :, :int(w * 0.75)]
    elif img.ndim == 3:
        _, w, _ = img.shape
        img_r = img[:, int(w * 0.25):, :]
        img_l = img[:, :int(w * 0.75), :]
    else:
        raise ValueError(f"Unsupported image ndim for laterality test: {img.ndim}")

    sum_r = float(np.sum(img_r))
    sum_l = float(np.sum(img_l))
    if direction == "r2l":
        return sum_r > sum_l
    if direction == "l2r":
        return sum_l > sum_r
    return False


def _horizontal_flip(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[:, ::-1].copy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        return img[:, :, ::-1].copy()
    if img.ndim == 3:
        return img[:, ::-1, :].copy()
    raise ValueError(f"Unsupported image ndim for horizontal flip: {img.ndim}")


def flip_image(
    image: Union[np.ndarray, str, Path],
    direction: Optional[Union[str, bool]] = "r2l",
    stack_channels: bool = False,
) -> np.ndarray:
    """Flip mammogram image horizontally if its laterality matches the trigger."""
    direction = normalize_flip_direction(direction)
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    if direction is not None and _laterality_should_flip(img, direction):
        img = _horizontal_flip(img)

    if stack_channels and img.ndim == 2:
        img = np.stack([img, img, img], axis=0)
    return img


def np_CountUpContinuingOnes(b_arr: np.ndarray) -> np.ndarray:
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]

    return right - left - 1


def np_ExtractBreast(img: np.ndarray) -> np.ndarray:
    """Crop the breast region from background artifacts."""
    img_copy = img.copy()
    img_thresh = np.where(img <= 40, 0, img)

    height, _ = img_thresh.shape
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr_y = img_thresh[y_b:y_a].std(axis=0) != 0
    continuing_ones_y = np_CountUpContinuingOnes(b_arr_y)
    col_ind = np.where(continuing_ones_y == continuing_ones_y.max())[0]
    img_thresh = img_thresh[:, col_ind]

    _, width = img_thresh.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr_x = img_thresh[:, x_b:x_a].std(axis=1) != 0
    continuing_ones_x = np_CountUpContinuingOnes(b_arr_x)
    row_ind = np.where(continuing_ones_x == continuing_ones_x.max())[0]

    return img_copy[row_ind][:, col_ind]


def read_array(img_path: Union[str, Path], direction: Optional[Union[str, bool]] = None) -> np.ndarray:
    """
    Read mammography image (DICOM or PNG/JPG), crop breast tissue, resize to (1520, 912),
    and optionally apply laterality flipping. Returns uint8 array of shape (1520, 912).
    """
    img_str = str(img_path)
    
    # Standard image formats (PNG, JPG)
    if img_str.lower().endswith(('.png', '.jpg', '.jpeg')):
        data = cv2.imread(img_str, cv2.IMREAD_UNCHANGED)
        if data is None:
            data = np.array(Image.open(img_str))

        if len(data.shape) == 3:
            if data.shape[2] == 3:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            elif data.shape[2] == 4:
                data = cv2.cvtColor(data, cv2.COLOR_BGRA2GRAY)
            else:
                data = data[:, :, 0]

        data = data.astype('float32')
        data = data - np.min(data)
        mx = float(np.max(data))
        data = (data / mx) if mx > 0 else np.zeros_like(data, dtype=np.float32)
        data = (data * 255).astype(np.uint8)

        if data.shape != (1520, 912):
            try:
                data = np_ExtractBreast(data)
            except Exception:
                pass
            data = cv2.resize(data, (912, 1520), interpolation=cv2.INTER_AREA)

        direction = normalize_flip_direction(direction)
        if direction is not None:
            data = flip_image(data, direction=direction)

        return data

    # DICOM format
    dicom = pydicom.dcmread(img_path)
    dcm_sdl = dicomsdl.open(img_path)
    try:
        data = dcm_sdl.pixelData()
    except Exception:
        data = dicom.pixel_array

    data = data[5:-5, 5:-5]
    try:
        data = pydicom.pixels.processing.apply_voi_lut(data, dicom)
    except Exception:
        pass

    if dcm_sdl.getPixelDataInfo().get('PhotometricInterpretation') == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    mx = float(np.max(data))
    data = (data / mx) if mx > 0 else np.zeros_like(data, dtype=np.float32)
    data = (data * 255).astype(np.uint8)

    if len(data.shape) == 3:
        img_id = data.shape[0] // 2
        data = data[img_id, :, :]

    data = np_ExtractBreast(data)
    data = cv2.resize(data, (912, 1520), interpolation=cv2.INTER_AREA)

    direction = normalize_flip_direction(direction)
    if direction is not None:
        data = flip_image(data, direction=direction)

    return data


# ==============================================================================
# DATASET & DATALOADER
# ==============================================================================
class ImageTextDataset(Dataset):
    """
    Dataset extracting 4 standard mammogram views and temporal clinical note segments.
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        df: pd.DataFrame,
        text_max_length: int = 512,
        max_segments: int = 45,
        scale: int = 30,
        direction: Optional[Union[str, bool]] = "r2l",
        mean: float = 0.3089209,
        std: float = 0.25053555208335154,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.max_segments = max_segments
        self.scale = scale
        self.mean = mean
        self.std = std
        self.direction = normalize_flip_direction(direction)
        self.views = ['CC_L', 'MLO_L', 'CC_R', 'MLO_R']

    def __len__(self) -> int:
        return len(self.df)

    def _process_image(self, img_path: Any) -> torch.Tensor:
        """Load, crop, resize, flip, and standardize a single mammogram view."""
        channels = 3
        if pd.isna(img_path):
            return torch.zeros((channels, 1520, 912), dtype=torch.float32)

        if isinstance(img_path, (list, tuple)):
            clean_path = str(img_path[0]) if len(img_path) > 0 else ""
        else:
            clean_path = str(img_path).strip()
            if clean_path.startswith('[') and clean_path.endswith(']'):
                inner = clean_path[1:-1].strip()
                clean_path = inner.split(',')[0].strip().strip("'\"") if ',' in inner else inner.strip("'\"")
            else:
                clean_path = clean_path.strip("'\"")

        if not clean_path:
            return torch.zeros((channels, 1520, 912), dtype=torch.float32)

        # Fall back between .dcm and .png if needed
        if not os.path.exists(clean_path):
            if clean_path.endswith('.dcm') and os.path.exists(clean_path[:-4] + '.png'):
                clean_path = clean_path[:-4] + '.png'
            elif clean_path.endswith('.png') and os.path.exists(clean_path[:-4] + '.dcm'):
                clean_path = clean_path[:-4] + '.dcm'

        if not os.path.exists(clean_path):
            return torch.zeros((channels, 1520, 912), dtype=torch.float32)

        try:
            image = read_array(clean_path, direction=self.direction)
            image = image.astype('float32') / 255.0
            image = (image - self.mean) / self.std
            image = np.stack((image,) * channels)
            return torch.tensor(image, dtype=torch.float32)
        except Exception as e:
            warnings.warn(f"Error processing image {clean_path}: {e}. Returning zero tensor.")
            return torch.zeros((channels, 1520, 912), dtype=torch.float32)

    def _process_text_timeline(self, text: Any) -> Tuple[List[str], List[float]]:
        """
        Parse visit-segmented clinical notes into (texts, time_deltas).
        Supports 'Date(X)' headers or falls back to line-by-line visits with delta=0.
        """
        if not text or not str(text).strip():
            return [], []

        segments = [s.strip() for s in str(text).split('\n') if s.strip()]
        if not segments:
            return [], []

        def _extract_date(tokens: List[str], scale: int = self.scale) -> float:
            if 'On' in tokens[0] or 'On' in tokens[:3]:
                return 0.0
            sign = len(tokens) > 2 and tokens[2] == 'after'
            num_str = tokens[0].replace('Date(', '').replace(')', '')
            try:
                num = float(num_str)
                return (num / scale) if sign else (0.0 - num / scale)
            except ValueError:
                return 0.0

        texts, time_deltas = [], []
        try:
            first_tokens = segments[0].split(' ')
            if len(first_tokens) >= 3 and ('Date(' in first_tokens[0] or 'On' in first_tokens[0]):
                base_date = _extract_date(first_tokens)
                for seg in segments:
                    tokens = seg.split(' ')
                    if len(tokens) < 3:
                        continue
                    date = _extract_date(tokens)
                    time_deltas.append(date - base_date)
                    if 'diagnosis):' in seg:
                        texts.append(seg.split('diagnosis):', 1)[1].strip())
                    elif 'date):' in seg:
                        texts.append(seg.split('date):', 1)[1].strip())
                    elif '):' in seg:
                        texts.append(seg.split('):', 1)[1].strip())
                    else:
                        texts.append(seg.strip())
            else:
                for i, seg in enumerate(segments):
                    texts.append(seg.strip())
                    time_deltas.append(float(i))
        except Exception:
            texts = [s.strip() for s in segments]
            time_deltas = [0.0] * len(texts)

        return texts, time_deltas

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]

        # Extract patient ID
        if 'PATIENT_CLINIC_NUMBER' in row and pd.notna(row['PATIENT_CLINIC_NUMBER']):
            patient_id = str(row['PATIENT_CLINIC_NUMBER'])
        elif 'patient_id' in row and pd.notna(row['patient_id']):
            patient_id = str(row['patient_id'])
        elif 'PatientID' in row and pd.notna(row['PatientID']):
            patient_id = str(row['PatientID'])
        else:
            patient_id = f"patient_{index}"

        # Extract time discrepancy in years
        time_val = row.get("Time_discrepancy", None)
        try:
            time_gaps = float(time_val) if pd.notna(time_val) else 0.0
        except (ValueError, TypeError):
            time_gaps = 0.0

        raw_text = str(row['summary']) if ('summary' in row and pd.notna(row['summary'])) else ""
        segment_texts, time_deltas = self._process_text_timeline(raw_text)

        images = {}
        for view in self.views:
            img_val = row.get(view, None)
            images[view] = self._process_image(img_val)

        return {
            "patient_id": patient_id,
            "time_discrepancy": time_gaps,
            "segment_texts": segment_texts,
            "time_deltas": time_deltas,
            "images": images,
        }

    def collate_fn(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate individual samples into GPU-ready batches."""
        batch_images = {
            view: torch.stack([ins["images"][view] for ins in instances], dim=0)
            for view in self.views
        }
        patient_ids = [ins["patient_id"] for ins in instances]
        time_gaps = torch.tensor(
            [ins["time_discrepancy"] for ins in instances], dtype=torch.float32
        )

        flat_texts = []
        time_deltas_padded = []
        padding_mask = []

        for ins in instances:
            texts = ins["segment_texts"][:self.max_segments]
            deltas = ins["time_deltas"][:self.max_segments]
            pad_len = self.max_segments - len(texts)
            flat_texts.extend(texts)
            flat_texts.extend([""] * pad_len)
            time_deltas_padded.append(deltas + [0.0] * pad_len)
            padding_mask.append([True] * len(texts) + [False] * pad_len)

        text_tokens = self.tokenizer(
            flat_texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.text_max_length,
        )

        batch_size = len(instances)
        for key in ['input_ids', 'attention_mask']:
            if key in text_tokens:
                seq_len = text_tokens[key].shape[1]
                text_tokens[key] = text_tokens[key].view(batch_size, self.max_segments, seq_len)

        return {
            "patient_ids": patient_ids,
            "time_discrepancy": time_gaps,
            "text_tokens": text_tokens,
            "time_deltas": torch.tensor(time_deltas_padded, dtype=torch.float32),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
            "images": batch_images,
        }


# ==============================================================================
# MODEL COMPONENTS: VIEW MEAN POOLING & TEMPORAL TEXT FUSION
# ==============================================================================
class ViewMeanPooling(nn.Module):
    """
    Mean pooling across mammogram views (CC_L, MLO_L, CC_R, MLO_R).
    Averages 4 view embeddings into a single visual embedding.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, view_seq=None) -> torch.Tensor:
        # x: [B, num_views, embed_dim] -> [B, embed_dim]
        return x.mean(dim=1)


class TemporalTextFusion(nn.Module):
    """
    Aggregates per-visit CLS embeddings using time-delta positional encoding
    and a shallow TransformerEncoder with a learnable CLS token.
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self,
        text_embeds: torch.Tensor,   # [B, N, D]
        time_deltas: torch.Tensor,   # [B, N]
        padding_mask: torch.Tensor,  # [B, N] True = real visit, False = padding
    ) -> torch.Tensor:               # [B, D]
        B, N, D = text_embeds.shape
        time_embeds = self.time_proj(time_deltas.unsqueeze(-1).float())
        x = text_embeds + time_embeds

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        ignore_mask = ~padding_mask
        full_ignore_mask = torch.cat((cls_mask, ignore_mask), dim=1)

        out = self.transformer(x, src_key_padding_mask=full_ignore_mask)
        return out[:, 0, :]


# ==============================================================================
# MULTIMODAL MODEL (Mean Pooling + Bidirectional Cross Attention)
# ==============================================================================
class MultimodalModel(nn.Module):
    """
    Inference model pairing ViewMeanPooling with Bidirectional Cross Attention (BCA).
    Extracts:
      - Unimodal visual embedding (2048-dim, 4-view mean-pooled)
      - Unimodal text embedding (768-dim, temporal aggregated)
      - Multimodal fused embedding (1024-dim, cross-attended and L2 normalized)
    """
    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        mammo_fm: Optional[Any] = None,
        fm_embed_dim: int = 2048,
        text_hidden_dim: int = 768,
        shared_dim: int = 1024,
        num_heads: int = 8,
        temporal_num_heads: int = 8,
        temporal_num_layers: int = 2,
        gate: Union[str, float, int] = "dynamic",
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.device = torch.device(device or DEVICE)
        self.gate = gate

        # Resolve image and text encoders
        if image_encoder is not None and text_encoder is not None:
            self.image_encoder = image_encoder
            self.text_encoder = text_encoder
        elif mammo_fm is not None:
            self.image_encoder = getattr(mammo_fm, "image_encoder", None)
            self.text_encoder = getattr(mammo_fm, "text_encoder", None)
        else:
            self.image_encoder = EfficientNet.from_name("efficientnet-b5", num_classes=1)
            self.text_encoder = AutoModel.from_pretrained(
                "answerdotai/ModernBERT-base",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )

        self.view_fusion = ViewMeanPooling()

        # Temporal textual fusion
        self.temporal_text_fusion = TemporalTextFusion(
            embed_dim=text_hidden_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
        )

        # Cross Attention Multimodal Fusion (BCA)
        self.visual_proj = nn.Linear(fm_embed_dim, shared_dim)
        self.text_proj = nn.Linear(text_hidden_dim, shared_dim)
        self.cross_attn_t2v = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_v2t = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=num_heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(shared_dim)

        self.to_devices()

    def to_devices(self):
        """Move all model components to target device."""
        self.to(self.device)

    def _encode_views(self, images_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode 4 views using EfficientNet-B5 with autocast."""
        view_order = ['CC_L', 'MLO_L', 'CC_R', 'MLO_R']
        img_embeds = []
        dev_type = self.device.type
        autocast_dtype = torch.bfloat16 if (dev_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        autocast_ctx = torch.autocast(device_type=dev_type, dtype=autocast_dtype) if dev_type == "cuda" else contextlib.nullcontext()

        with torch.no_grad(), autocast_ctx:
            for view in view_order:
                img = images_dict[view].to(self.device, non_blocking=False)
                img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                feat = self.image_encoder(img)
                img_embeds.append(feat)

        stacked = torch.stack(img_embeds, dim=1).float()
        stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e4, neginf=-1e4)
        stacked = F.normalize(stacked, p=2, dim=-1, eps=1e-6)
        return stacked

    def _encode_visits(self, text_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode clinical text visits using ModernBERT with autocast."""
        text_tokens = {k: v.to(self.device, non_blocking=False) for k, v in text_tokens.items()}
        B, N, SeqLen = text_tokens["input_ids"].shape
        flat_input_ids = text_tokens["input_ids"].view(B * N, SeqLen)
        flat_attention_mask = text_tokens["attention_mask"].view(B * N, SeqLen)

        dev_type = self.device.type
        autocast_dtype = torch.bfloat16 if (dev_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        autocast_ctx = torch.autocast(device_type=dev_type, dtype=autocast_dtype) if dev_type == "cuda" else contextlib.nullcontext()

        with torch.no_grad(), autocast_ctx:
            text_outputs = self.text_encoder(
                input_ids=flat_input_ids,
                attention_mask=flat_attention_mask,
            )
            flat_cls = (
                text_outputs.last_hidden_state[:, 0, :]
                if hasattr(text_outputs, 'last_hidden_state')
                else text_outputs[0]
            )

        segment_embeds = flat_cls.float().view(B, N, -1)
        segment_embeds = F.normalize(segment_embeds, p=2, dim=-1)
        return segment_embeds

    def forward(
        self,
        images_dict: Dict[str, torch.Tensor],
        text_tokens: Dict[str, torch.Tensor],
        time_gaps: torch.Tensor,
        time_deltas: torch.Tensor,
        padding_mask: torch.Tensor,
        return_unimodal: bool = True,
        gate: Optional[Union[str, float, int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass computing multimodal and unimodal embeddings.

        Returns:
            If return_unimodal is True:
                ranking_embed : (B, 1024) Multimodal cross-attended embedding (L2 normalized)
                visual_embed  : (B, 2048) Unimodal 4-view mean-pooled image embedding (L2 normalized)
                fused_text_seq: (B, 768)  Unimodal temporal text embedding (L2 normalized)
            Else:
                ranking_embed : (B, 1024)
        """
        # 1. Visual pipeline: 4 views -> Mean Pooling -> [B, 2048]
        stacked_img_embeds = self._encode_views(images_dict)
        fused_img_seq = self.view_fusion(stacked_img_embeds)

        # 2. Text pipeline: ModernBERT visits -> TemporalTextFusion -> [B, 768]
        segment_embeds = self._encode_visits(text_tokens)
        time_deltas = time_deltas.to(self.device, non_blocking=False)
        padding_mask = padding_mask.to(self.device, non_blocking=False)

        fused_text_seq = self.temporal_text_fusion(segment_embeds, time_deltas, padding_mask)
        fused_text_seq = torch.nan_to_num(fused_text_seq.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        fused_text_seq = F.normalize(fused_text_seq, p=2, dim=-1, eps=1e-6)

        # 3. Multimodal projection into shared dimension (1024)
        v_shared = self.visual_proj(fused_img_seq).unsqueeze(1)
        t_shared = self.text_proj(fused_text_seq).unsqueeze(1)

        # 4. Time-discrepancy cross-attention gating
        eff_gate = self.gate if gate is None else gate
        time_gaps = time_gaps.to(self.device, non_blocking=False).float()
        time_gaps = torch.nan_to_num(time_gaps, nan=0.0, posinf=1e3, neginf=-1e3)

        if (isinstance(eff_gate, str) and eff_gate.strip().lower() in ("static", "1", "one")) or (
            isinstance(eff_gate, (int, float)) and not isinstance(eff_gate, bool) and float(eff_gate) == 1.0
        ):
            gate_tensor = torch.ones((time_gaps.shape[0], 1, 1), dtype=torch.float32, device=self.device)
        elif isinstance(eff_gate, (int, float)) and not isinstance(eff_gate, bool):
            gate_tensor = torch.full((time_gaps.shape[0], 1, 1), float(eff_gate), dtype=torch.float32, device=self.device)
        else:
            gate_vals = torch.where(
                time_gaps > 1.0,
                torch.zeros_like(time_gaps),
                1.0 - time_gaps / 2.0,
            )
            gate_vals = torch.clamp(gate_vals, min=0.0, max=1.0)
            gate_tensor = gate_vals.unsqueeze(-1).unsqueeze(-1)

        v_value_gated = v_shared * gate_tensor

        # 5. Bidirectional cross attention (BCA)
        attn_t2v, _ = self.cross_attn_t2v(t_shared, v_value_gated, v_value_gated, need_weights=False)
        attn_v2t, _ = self.cross_attn_v2t(v_value_gated, t_shared, t_shared, need_weights=False)
        attn_v2t = attn_v2t * gate_tensor

        multimodal_embed = self.fusion_norm(t_shared.squeeze(1) + attn_t2v.squeeze(1) + attn_v2t.squeeze(1))
        ranking_embed = F.normalize(multimodal_embed, p=2, dim=-1)

        if return_unimodal:
            visual_embed = F.normalize(fused_img_seq, p=2, dim=-1)
            return ranking_embed, visual_embed, fused_text_seq

        return ranking_embed


# ==============================================================================
# MODEL LOADING UTILITIES
# ==============================================================================
def build_base_encoders() -> Tuple[nn.Module, nn.Module]:
    """
    Directly instantiate the image encoder (EfficientNet-B5) and text encoder (ModernBERT)
    without loading the 2.1 GB Mammo-FM_ASU_Trained_CLIP.tar base checkpoint.
    All weights are populated directly from the multimodal .pth checkpoint.
    """
    image_encoder = EfficientNet.from_name("efficientnet-b5", num_classes=1)
    text_encoder = AutoModel.from_pretrained(
        "answerdotai/ModernBERT-base",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    return image_encoder, text_encoder


def build_model_and_tokenizer(
    base_ckpt_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    gate: Union[str, float, int] = "dynamic",
) -> Tuple[AutoTokenizer, "MultimodalModel"]:
    """
    Instantiate tokenizer and MultimodalModel skeleton.
    If base_ckpt_path is provided and exists, reads from that file;
    otherwise initializes the architecture directly without reading the base .tar.
    """
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    if base_ckpt_path and os.path.exists(base_ckpt_path):
        print(f"[Inference] Initializing architecture using base checkpoint: {base_ckpt_path}...")
        _, mammo_fm = load_base_model(checkpoint_path=base_ckpt_path)
        model = MultimodalModel(mammo_fm=mammo_fm, gate=gate, device=device)
    else:
        print("[Inference] Initializing model directly (without loading Mammo-FM_ASU_Trained_CLIP.tar)...")
        image_encoder, text_encoder = build_base_encoders()
        model = MultimodalModel(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            gate=gate,
            device=device,
        )
    return tokenizer, model


def load_base_model(checkpoint_path: Optional[str] = None) -> Tuple[AutoTokenizer, nn.Module]:
    """
    Load base Mammo-FM CLIP architecture and tokenizer from checkpoint file (legacy fallback).
    """
    resolved_path = checkpoint_path or resolve_base_checkpoint()
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Mammo-FM base checkpoint not found at: {resolved_path}")

    print(f"Reading base checkpoint file: {resolved_path}...")
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)

    model_config = checkpoint["config"]["model"]
    loss_config = checkpoint["config"]["loss"]

    if "text_encoder" in model_config and "cache_dir" in model_config["text_encoder"]:
        model_config["text_encoder"]["cache_dir"] = "answerdotai/ModernBERT-base"

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    if build_model is None:
        raise ImportError("build_model could not be imported from breastclip.model.")
    model = build_model(model_config=model_config, loss_config=loss_config, tokenizer=tokenizer)

    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=False)

    model = model.to("cpu")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return tokenizer, model


# ==============================================================================
# EVALUATION & EMBEDDING EXTRACTION
# ==============================================================================
def evaluate_model(
    test_csv_path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    ckpt_path: Optional[str] = DEFAULT_MULTIMODAL_CKPT,
    direction: Optional[Union[str, bool]] = "r2l",
    batch_size: int = 4,
    gate: Union[str, float, int] = "dynamic",
    time_scale: int = 30,
    device: Optional[Union[str, torch.device]] = None,
    output_path: Optional[str] = None,
    num_workers: int = 2,
    base_ckpt_path: Optional[str] = None,
) -> Dict[str, Union[List[str], np.ndarray]]:
    """
    Run inference to extract unimodal visual, unimodal text, and multimodal embeddings.

    Args:
        test_csv_path : Path to CSV file containing patient data.
        df            : Optional pandas DataFrame (used if test_csv_path is not provided).
        ckpt_path     : Path to multimodal checkpoint. Defaults to:
                        checkpoint/Emory_BCA_True_fp16_frozen_timeFilter+decay_MeanPooling_bs48_flipr2l_final16SIM/...
        direction     : Optional laterality flipping direction ('r2l', 'l2r', 'none', None).
        batch_size    : DataLoader batch size (default: 4).
        gate          : Gating mode for cross-attention ('dynamic' or 'static' / 1.0).
        time_scale    : Divisor for days in clinical notes timeline (default: 30).
        device        : Torch device or str ('cuda:0', 'cpu', etc.).
        output_path   : Optional path to save extracted embeddings (.npz or .pt).
        num_workers   : Number of worker processes for DataLoader (default: 2).
        base_ckpt_path: Optional path to base Mammo-FM .tar checkpoint.

    Returns:
        Dict with keys:
          - 'patient_ids': list of patient ID strings (N,)
          - 'multimodal_embeddings': np.ndarray (N, 1024)
          - 'visual_embeddings': np.ndarray (N, 2048)
          - 'text_embeddings': np.ndarray (N, 768)
          - 'multimodal': alias for 'multimodal_embeddings'
          - 'visual': alias for 'visual_embeddings'
          - 'text': alias for 'text_embeddings'
    """
    norm_direction = normalize_flip_direction(direction)
    flip_tag = norm_direction if norm_direction is not None else "none"

    # Setup device
    target_device = torch.device(device) if device is not None else DEVICE
    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        print(f"[Inference] Running on GPU: {target_device} ({torch.cuda.get_device_name(target_device)})")
    else:
        print(f"[Inference] Running on {target_device}")

    # Load input dataframe
    if df is None:
        if test_csv_path is None:
            raise ValueError("Either 'test_csv_path' or 'df' must be provided.")
        resolved_csv = resolve_file_path(test_csv_path)
        if not os.path.exists(resolved_csv):
            raise FileNotFoundError(f"Input CSV file not found at: {resolved_csv}")
        print(f"[Inference] Loading CSV: {resolved_csv}")
        df = pd.read_csv(resolved_csv)
    else:
        print(f"[Inference] Using provided DataFrame with {len(df)} records.")

    # 1. Build MultimodalModel and tokenizer directly without requiring base .tar
    tokenizer, model = build_model_and_tokenizer(
        base_ckpt_path=base_ckpt_path,
        device=target_device,
        gate=gate,
    )

    # 3. Load Multimodal Checkpoint
    resolved_ckpt = resolve_file_path(ckpt_path) if ckpt_path else None
    if resolved_ckpt and os.path.exists(resolved_ckpt):
        print(f"[Inference] Loading weights from checkpoint: {resolved_ckpt}")
        ckpt_dict = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
        if isinstance(ckpt_dict, dict) and "state_dict" in ckpt_dict:
            ckpt_dict = ckpt_dict["state_dict"]
        elif isinstance(ckpt_dict, dict) and "model" in ckpt_dict:
            ckpt_dict = ckpt_dict["model"]
        cleaned_weights = {k.replace("module.", ""): v for k, v in ckpt_dict.items()}
        load_res = model.load_state_dict(cleaned_weights, strict=True)
        print(f"[Inference] Checkpoint successfully loaded: {load_res}")
    else:
        if ckpt_path:
            warnings.warn(f"Checkpoint not found at {resolved_ckpt or ckpt_path}. Using base weights!")

    model.to_devices()
    model.eval()

    # 4. Prepare Dataset & DataLoader
    dataset = ImageTextDataset(
        tokenizer=tokenizer,
        df=df,
        scale=time_scale,
        direction=norm_direction,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )

    print(f"[Inference] Extracting embeddings for {len(dataset)} patients (direction={flip_tag}, batch_size={batch_size})...")

    patient_ids_list = []
    all_fused = []
    all_visual = []
    all_text = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            fused_emb, visual_emb, text_emb = model(
                images_dict=batch["images"],
                text_tokens=batch["text_tokens"],
                time_gaps=batch["time_discrepancy"],
                time_deltas=batch["time_deltas"],
                padding_mask=batch["padding_mask"],
                return_unimodal=True,
                gate=gate,
            )
            patient_ids_list.extend(batch["patient_ids"])
            all_fused.append(fused_emb.detach().float().cpu().numpy())
            all_visual.append(visual_emb.detach().float().cpu().numpy())
            all_text.append(text_emb.detach().float().cpu().numpy())

    fused_embeddings = np.concatenate(all_fused, axis=0) if all_fused else np.empty((0, 1024))
    visual_embeddings = np.concatenate(all_visual, axis=0) if all_visual else np.empty((0, 2048))
    text_embeddings = np.concatenate(all_text, axis=0) if all_text else np.empty((0, 768))

    results = {
        "patient_ids": patient_ids_list,
        "multimodal_embeddings": fused_embeddings,
        "visual_embeddings": visual_embeddings,
        "text_embeddings": text_embeddings,
        # Convenient aliases
        "multimodal": fused_embeddings,
        "visual": visual_embeddings,
        "text": text_embeddings,
    }

    print(
        f"[Inference] Completed! Extracted shapes: "
        f"Multimodal {fused_embeddings.shape}, "
        f"Visual {visual_embeddings.shape}, "
        f"Text {text_embeddings.shape}"
    )

    # 5. Optional file saving
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix in (".pt", ".pth"):
            save_payload = {
                "patient_ids": patient_ids_list,
                "multimodal_embeddings": torch.from_numpy(fused_embeddings),
                "visual_embeddings": torch.from_numpy(visual_embeddings),
                "text_embeddings": torch.from_numpy(text_embeddings),
            }
            torch.save(save_payload, str(out_p))
            print(f"[Inference] Saved PyTorch tensors to: {out_p}")
        else:
            # Default to npz compressed
            if not out_p.suffix:
                out_p = out_p.with_suffix(".npz")
            np.savez_compressed(
                str(out_p),
                patient_ids=np.array(patient_ids_list),
                multimodal_embeddings=fused_embeddings,
                visual_embeddings=visual_embeddings,
                text_embeddings=text_embeddings,
            )
            print(f"[Inference] Saved compressed NumPy archive to: {out_p}")

    return results


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract unimodal & multimodal embeddings using ViewMeanPooling + Bidirectional Cross Attention."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=resolve_default_csv(),
        help="Path to input CSV file (default: data/test_cohort.csv or data_csv/...).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=DEFAULT_MULTIMODAL_CKPT,
        help="Path to trained MultimodalModel checkpoint (.pth).",
    )
    parser.add_argument(
        "--base_ckpt",
        type=str,
        default=None,
        help="Path to base Mammo-FM .tar checkpoint (optional).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="r2l",
        choices=["r2l", "l2r", "none"],
        help="Optional laterality flipping direction ('r2l', 'l2r', or 'none'). Default: 'r2l'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--gate",
        type=str,
        default="dynamic",
        help="Gate mode: 'dynamic' (time-discrepancy decay) or 'static' (gate=1.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extracted_embeddings.npz",
        help="Path to output file (.npz or .pt).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker processes.",
    )

    args = parser.parse_args()

    evaluate_model(
        test_csv_path=args.csv,
        ckpt_path=args.ckpt,
        direction=args.direction,
        batch_size=args.batch_size,
        gate=args.gate,
        output_path=args.output,
        num_workers=args.num_workers,
        base_ckpt_path=args.base_ckpt,
    )


if __name__ == "__main__":
    main()
