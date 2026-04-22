"""
Unit tests for ESG multi-task classification components.
"""

import sys, os
import unittest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config
from utils.dataset import encode_labels, normalise_field
from models.model import ESGMultiTaskModel


class TestLabelMapping(unittest.TestCase):
    """Verify label encoding handles all edge cases."""

    def test_commitment_yes(self):
        sample = {"promise_status": "Yes"}
        labels = encode_labels(sample)
        self.assertEqual(labels["commitment"], 1)

    def test_commitment_no(self):
        sample = {"promise_status": "No"}
        labels = encode_labels(sample)
        self.assertEqual(labels["commitment"], 0)

    def test_empty_string_maps_to_ignore(self):
        sample = {
            "promise_status": "Yes",
            "evidence_status": "",
            "evidence_quality": "",
            "verification_timeline": "",
        }
        labels = encode_labels(sample)
        self.assertEqual(labels["evidence"], config.IGNORE_INDEX)
        self.assertEqual(labels["clarity"], config.IGNORE_INDEX)
        self.assertEqual(labels["timeline"], config.IGNORE_INDEX)

    def test_na_maps_to_ignore(self):
        sample = {
            "promise_status": "No",
            "evidence_status": "N/A",
            "evidence_quality": "N/A",
            "verification_timeline": "N/A",
        }
        labels = encode_labels(sample)
        self.assertEqual(labels["evidence"], config.IGNORE_INDEX)
        self.assertEqual(labels["clarity"], config.IGNORE_INDEX)
        self.assertEqual(labels["timeline"], config.IGNORE_INDEX)

    def test_clarity_misleading(self):
        sample = {"evidence_quality": "Misleading"}
        labels = encode_labels(sample)
        self.assertEqual(labels["clarity"], 2)

    def test_timeline_between_2_and_5(self):
        sample = {"verification_timeline": "between_2_and_5_years"}
        labels = encode_labels(sample)
        self.assertEqual(labels["timeline"], 2)

    def test_normalise_field_none(self):
        self.assertEqual(normalise_field(None), "")

    def test_missing_field_defaults_to_ignore(self):
        labels = encode_labels({})
        for task in config.TASK_NAMES:
            self.assertEqual(labels[task], config.IGNORE_INDEX)


class TestModel(unittest.TestCase):
    """Smoke test the model forward pass."""

    def test_output_shapes(self):
        model = ESGMultiTaskModel()
        model.eval()
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        self.assertIsInstance(logits, dict)
        for task in config.TASK_NAMES:
            self.assertIn(task, logits)
            self.assertEqual(logits[task].shape,
                             (batch_size, config.NUM_CLASSES[task]))


if __name__ == "__main__":
    unittest.main()
