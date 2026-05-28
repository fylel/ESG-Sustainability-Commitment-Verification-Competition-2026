"""
Multi-task BERT classifier for ESG classification.

Architecture
────────────
  [CLS] text tokens [SEP] [PAD]…
        │
   BERT encoder (shared)
        │
   [CLS] hidden  (768-d)
        │
   ┌────┴────┬──────────┬──────────┐
   head_0   head_1    head_2     head_3
 commitment evidence  clarity   timeline
   (2)       (2)       (3)       (4)
"""

import torch
import torch.nn as nn
from transformers import AutoModel

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config


class TaskHead(nn.Module):
    """A single classification head: dropout → linear."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(cls_hidden))


class SpanHead(nn.Module):
    """Predicts start/end token positions of a text span.

    Operates on the full token sequence (not just [CLS]) so the model
    learns to locate specific spans within the input text.
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 2)  # column 0 = start, 1 = end

    def forward(self, sequence_output: torch.Tensor):
        # sequence_output: (B, seq_len, hidden_dim)
        out = self.linear(self.dropout(sequence_output))  # (B, seq_len, 2)
        return out[..., 0], out[..., 1]  # start_logits, end_logits each (B, seq_len)


class KeywordHead(nn.Module):
    """Binary token classifier: keyword (1) vs non-keyword (0)."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(sequence_output))  # (B, seq_len, 2)


class ESGMultiTaskModel(nn.Module):
    """
    Shared BERT encoder + one TaskHead per task.

    forward() returns a dict  {task_name: logits}  where logits shape is
    (batch_size, num_classes_for_task).
    """

    def __init__(
        self,
        pretrained: str = config.PRETRAINED_MODEL,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = config.CLASSIFIER_DROPOUT,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained)

        self.heads = nn.ModuleDict({
            task: TaskHead(hidden_dim, n_cls, dropout)
            for task, n_cls in config.NUM_CLASSES.items()
        })

        if config.USE_SPAN_AUX:
            self.promise_span_head = SpanHead(hidden_dim, dropout)
            self.evidence_span_head = SpanHead(hidden_dim, dropout)

        if config.USE_KEYWORD_AUX:
            self.keyword_head = KeywordHead(hidden_dim, dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state.float()  # (B, seq_len, hidden)
        cls_hidden = sequence_output[:, 0, :]                # [CLS] vector

        logits = {
            task: head(cls_hidden)
            for task, head in self.heads.items()
        }

        if config.USE_SPAN_AUX:
            logits["promise_start"], logits["promise_end"] = \
                self.promise_span_head(sequence_output)
            logits["evidence_start"], logits["evidence_end"] = \
                self.evidence_span_head(sequence_output)

        if config.USE_KEYWORD_AUX:
            logits["keyword"] = self.keyword_head(sequence_output)  # (B, seq_len, 2)

        return logits
