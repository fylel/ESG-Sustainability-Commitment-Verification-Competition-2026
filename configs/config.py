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

IGNORE_INDEX = -1  # label value for samples excluded from a task's loss

COMMITMENT_MAP = {"No": 0, "Yes": 1}
EVIDENCE_MAP = {"No": 0, "Yes": 1, "N/A": IGNORE_INDEX, "": IGNORE_INDEX}
CLARITY_MAP = {
    "Clear": 0, "Not Clear": 1, "Misleading": 2,
    "N/A": IGNORE_INDEX, "": IGNORE_INDEX,
}
TIMELINE_MAP = {
    "already": 0,
    "within_2_years": 1,
    "between_2_and_5_years": 2,
    "more_than_5_years": 3,
    "N/A": IGNORE_INDEX,
    "": IGNORE_INDEX,
}

LABEL_MAPS = {
    "commitment": COMMITMENT_MAP,
    "evidence": EVIDENCE_MAP,
    "clarity": CLARITY_MAP,
    "timeline": TIMELINE_MAP,
}

# Number of real classes per task (excluding IGNORE_INDEX)
NUM_CLASSES = {
    "commitment": 2,
    "evidence": 2,
    "clarity": 3,
    "timeline": 4,
}

# ─── Model ────────────────────────────────────────────────────────────
PRETRAINED_MODEL = "hfl/chinese-roberta-wwm-ext"
MAX_SEQ_LEN = 256
HIDDEN_DIM = 768          # must match pretrained encoder
CLASSIFIER_DROPOUT = 0.1

# ─── Training ─────────────────────────────────────────────────────────
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.1
SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── Task Loss Weights (for combined loss) ────────────────────────────
TASK_LOSS_WEIGHTS = {
    "commitment": 1.0,
    "evidence": 1.0,
    "clarity": 1.0,
    "timeline": 1.0,
}
