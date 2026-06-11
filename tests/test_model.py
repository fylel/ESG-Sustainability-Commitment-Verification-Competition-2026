"""
Unit tests for ESG multi-task classification components.
"""

import sys, os
import unittest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config
from utils.dataset import (
    encode_labels, normalise_field,
    _parse_year, _phrase_chars_in_span,
    _ALREADY_PHRASES, _RELATIVE_TIME_PHRASES,
)
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


class TestParseYear(unittest.TestCase):

    def test_valid_years(self):
        for yr in ("1990", "2025", "2030", "2050", "2100"):
            with self.subTest(yr=yr):
                self.assertEqual(_parse_year(yr), int(yr))

    def test_boundary_invalid(self):
        self.assertEqual(_parse_year("1989"), -1)
        self.assertEqual(_parse_year("2101"), -1)

    def test_short_numbers_rejected(self):
        for tok in ("30", "203", "25"):
            with self.subTest(tok=tok):
                self.assertEqual(_parse_year(tok), -1)

    def test_non_numeric_rejected(self):
        for tok in ("年", "abc", "20%", "", " "):
            with self.subTest(tok=tok):
                self.assertEqual(_parse_year(tok), -1)


class TestPhraseCharsInSpan(unittest.TestCase):

    def test_phrase_inside_span(self):
        text = "公司已完成碳中和目標"
        span = set(range(len(text)))
        result = _phrase_chars_in_span(text, ("已完成",), span)
        self.assertEqual(result, {2, 3, 4})

    def test_phrase_outside_span_ignored(self):
        text = "已完成目標，計畫五年內完成新目標"
        span = set(range(6, len(text)))   # only second half
        result = _phrase_chars_in_span(text, ("已完成",), span)
        self.assertEqual(result, set())

    def test_multiple_occurrences_both_captured(self):
        text = "已完成A，已完成B"
        span = set(range(len(text)))
        result = _phrase_chars_in_span(text, ("已完成",), span)
        self.assertEqual(result, {0, 1, 2, 5, 6, 7})

    def test_relative_phrase_in_span(self):
        text = "承諾五年內達成碳中和"
        span = set(range(len(text)))
        result = _phrase_chars_in_span(text, _RELATIVE_TIME_PHRASES, span)
        # "五年內" starts at index 2
        self.assertTrue({2, 3, 4}.issubset(result))


class TestTemporalLabelLogic(unittest.TestCase):
    """
    Simulate macbert char-level tokenisation (Chinese chars = 1 token,
    4-digit year = 1 token) and verify temporal label assignment.
    """

    def _make_offsets(self, text, max_seq_len=64):
        """Build mock offset_mapping for a text.
        Treats 4-consecutive-digit runs as single tokens (macbert behaviour).
        Returns list of (start, end) char offsets, length == max_seq_len.
        """
        import re
        tokens = []
        i = 0
        while i < len(text):
            # 4-digit year: consume as one token
            if re.match(r'\d{4}', text[i:i+4]) and (i+4 >= len(text) or not text[i+4].isdigit()):
                tokens.append((i, i + 4))
                i += 4
            else:
                tokens.append((i, i + 1))
                i += 1

        offsets = [(0, 0)]                 # [CLS]
        offsets += tokens[:max_seq_len - 2]
        offsets += [(0, 0)]                # [SEP]
        while len(offsets) < max_seq_len:
            offsets.append((0, 0))        # PAD
        return offsets[:max_seq_len]

    def _run(self, text, promise, timeline_str, max_seq_len=64):
        """Run the temporal label logic from dataset.py using mock offsets."""
        tl_map = config.TIMELINE_MAP
        timeline_label = tl_map.get(timeline_str, -1)
        near = config.TIMELINE_NEAR_CLASSES
        far  = config.TIMELINE_FAR_CLASSES

        if timeline_label not in near and timeline_label not in far:
            return [-1] * max_seq_len

        target_cls = 1 if timeline_label in near else 2
        offsets = self._make_offsets(text, max_seq_len)

        span_char_set: set = set()
        cs = text.find(promise)
        if cs != -1:
            span_char_set.update(range(cs, cs + len(promise)))

        max_year = -1
        for s, e in offsets:
            if e > s and any(ci in span_char_set for ci in range(s, e)):
                y = _parse_year(text[s:e])
                if y > max_year:
                    max_year = y

        relative_chars = _phrase_chars_in_span(text, _RELATIVE_TIME_PHRASES, span_char_set)
        already_chars: set = set()
        if timeline_label == 0 and max_year == -1:
            already_chars = _phrase_chars_in_span(text, _ALREADY_PHRASES, span_char_set)

        result = []
        for s, e in offsets:
            if e <= s:
                result.append(-1)
            elif not (span_char_set and any(ci in span_char_set for ci in range(s, e))):
                result.append(-1)
            else:
                tok_chars = set(range(s, e))
                y = _parse_year(text[s:e])
                if y != -1:
                    result.append(target_cls if y == max_year else 0)
                elif tok_chars & relative_chars:
                    result.append(target_cls)
                elif tok_chars & already_chars:
                    result.append(1)
                else:
                    result.append(0)
        return result

    def _label_at_char(self, labels, offsets, char_pos):
        """Return the label of the token that covers char_pos."""
        for i, (s, e) in enumerate(offsets):
            if e > s and s <= char_pos < e:
                return labels[i]
        return None

    # ── test cases ────────────────────────────────────────────────────

    def test_baseline_year_zero_target_year_labeled(self):
        text = "公司以2020年為基準，承諾2030年前達成碳中和。"
        promise = "以2020年為基準，承諾2030年前達成碳中和。"
        labels = self._run(text, promise, "more_than_5_years")
        offsets = self._make_offsets(text)

        self.assertEqual(self._label_at_char(labels, offsets, text.index("2030")), 2)  # long-term
        self.assertEqual(self._label_at_char(labels, offsets, text.index("2020")), 0)  # baseline → 0

    def test_single_year_within2_labeled_near(self):
        text = "預計於2025年完成再生能源轉型。"
        promise = "預計於2025年完成再生能源轉型。"
        labels = self._run(text, promise, "within_2_years")
        offsets = self._make_offsets(text)
        self.assertEqual(self._label_at_char(labels, offsets, text.index("2025")), 1)

    def test_already_phrase_labeled_when_no_year(self):
        text = "本公司已完成碳中和目標，領先業界。"
        promise = "已完成碳中和目標，領先業界。"
        labels = self._run(text, promise, "already")
        # "已完成" starts at text.index("已完成")
        pos = text.index("已完成")
        offsets = self._make_offsets(text)
        self.assertEqual(self._label_at_char(labels, offsets, pos), 1)

    def test_already_phrase_suppressed_when_year_present(self):
        text = "本公司已完成2023年碳中和目標。"
        promise = "已完成2023年碳中和目標。"
        labels = self._run(text, promise, "already")
        offsets = self._make_offsets(text)
        # max_year=2023 present → already_chars empty → "已完成" chars get 0
        pos = text.index("已完成")
        self.assertEqual(self._label_at_char(labels, offsets, pos), 0)
        # 2023 is the only year → labeled near-term=1
        self.assertEqual(self._label_at_char(labels, offsets, text.index("2023")), 1)

    def test_relative_phrase_labeled(self):
        text = "公司承諾五年內達成RE100目標。"
        promise = "承諾五年內達成RE100目標。"
        labels = self._run(text, promise, "between_2_and_5_years")
        offsets = self._make_offsets(text)
        pos = text.index("五年內")
        self.assertEqual(self._label_at_char(labels, offsets, pos), 2)

    def test_outside_span_is_minus_one(self):
        text = "前言。公司承諾2030年達成目標。"
        promise = "公司承諾2030年達成目標。"
        labels = self._run(text, promise, "more_than_5_years")
        offsets = self._make_offsets(text)
        # "前言。" is outside span → should all be -1
        for s, e in offsets:
            if e > s and e <= text.index("公"):
                idx = offsets.index((s, e))
                self.assertEqual(labels[idx], -1)

    def test_na_timeline_all_minus_one(self):
        text = "本公司無具體承諾。"
        promise = "無具體承諾。"
        labels = self._run(text, promise, "N/A")
        self.assertTrue(all(l == -1 for l in labels))


if __name__ == "__main__":
    unittest.main()
