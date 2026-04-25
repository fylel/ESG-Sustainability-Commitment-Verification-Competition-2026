"""
VeriPromiseESG4K — ESG Multi-task Classification Config
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────
DATASET_NAME = "VeriPromiseESG4K"
MAX_SAMPLES = 1000
TEXT_FIELD = "data"

USED_FIELDS = ["data", "promise_status", "verification_timeline",
               "evidence_status", "evidence_quality"]

# ─── Tasks & Label Mappings ──────────────────────────────────────────
TASK_NAMES = ["commitment", "evidence", "clarity", "timeline"]

TASK_SOURCE_FIELDS = {
    "commitment": "promise_status",
    "evidence": "evidence_status",
    "clarity": "evidence_quality",
    "timeline": "verification_timeline",
}

IGNORE_INDEX = -1  # only for truly missing labels (empty string)

COMMITMENT_MAP = {"No": 0, "Yes": 1, "": IGNORE_INDEX}
EVIDENCE_MAP   = {"No": 0, "Yes": 1, "N/A": 2, "": IGNORE_INDEX}
CLARITY_MAP    = {"Clear": 0, "Not Clear": 1, "Misleading": 2, "N/A": 3, "": IGNORE_INDEX}
TIMELINE_MAP   = {
    "already": 0,
    "within_2_years": 1,
    "between_2_and_5_years": 2,
    "more_than_5_years": 3,
    "N/A": 4,
    "": IGNORE_INDEX,
}

LABEL_MAPS = {
    "commitment": COMMITMENT_MAP,
    "evidence": EVIDENCE_MAP,
    "clarity": CLARITY_MAP,
    "timeline": TIMELINE_MAP,
}

# N/A is now a real class, so class counts increase
NUM_CLASSES = {
    "commitment": 2,
    "evidence": 3,
    "clarity": 4,
    "timeline": 5,
}

# ─── Model ────────────────────────────────────────────────────────────
PRETRAINED_MODEL = "hfl/chinese-macbert-base"
MAX_SEQ_LEN = 256
HIDDEN_DIM = 768          # must match pretrained encoder
CLASSIFIER_DROPOUT = 0.1

# ─── Training ─────────────────────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 3
SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── Task Loss Weights (for combined loss) ────────────────────────────
TASK_LOSS_WEIGHTS = {
    "commitment": 0.20,
    "evidence": 0.30,
    "clarity": 0.35,
    "timeline": 0.15,
}

# ─── Competition Evaluation Weights ──────────────────────────────────
EVAL_WEIGHTS = {
    "commitment": 0.20,     # promise_status
    "timeline": 0.15,       # verification_timeline
    "evidence": 0.30,       # evidence_status
    "clarity": 0.35,        # evidence_quality
}
