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
from transformers import BertModel

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
        self.encoder = BertModel.from_pretrained(pretrained)

        self.heads = nn.ModuleDict({
            task: TaskHead(hidden_dim, n_cls, dropout)
            for task, n_cls in config.NUM_CLASSES.items()
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] vector

        logits = {
            task: head(cls_hidden)
            for task, head in self.heads.items()
        }
        return logits
