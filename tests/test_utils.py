import ast
import json
import unittest
from pathlib import Path

from src.utils import extract_structured


ROOT = Path(__file__).resolve().parents[1]


class ExtractStructuredTests(unittest.TestCase):
    def test_invoice_debug_sample_keeps_keys_values_and_multiline_blocks_aligned(self):
        with open(ROOT / "debug" / "model" / "2026-597_1.json") as handle:
            labeled_words = json.load(handle)

        words = [
            {
                "text": item["words"],
                "bbox": ast.literal_eval(item["bbox"]),
                "label": item["label"],
            }
            for item in labeled_words
        ]

        structured = extract_structured(words)

        self.assertEqual(structured["GENERAL"]["Request ID:"], "2026-597")
        self.assertEqual(structured["GENERAL"]["Trip Provider:"], "Sovereign Escapes")
        self.assertNotIn("Request", structured["GENERAL"])
        self.assertNotIn("Passenger 1", structured["PASSENGER INFORMATION"])
        self.assertEqual(structured["PASSENGER INFORMATION"]["Last Name:"], "Jashwanthwe")
        self.assertEqual(structured["PASSENGER INFORMATION"]["First Name:"], "Naga")
        self.assertEqual(structured["BILLING INFORMATION"]["Part 1:"], "0.00 miles X $0.525/mile = $00.00")
        self.assertEqual(structured["BILLING INFORMATION"]["Part 1: (2)"], "X $47.43/hour = $11.86")
        self.assertEqual(structured["TRANSPORT SUMMARY - PART 1"]["Person/Agency"], "frcfdcd edfefr")
        self.assertEqual(structured["TRANSPORT SUMMARY - PART 1"]["Person/Agency (2)"], "fdfcd fddfcd")
        self.assertIn(
            "Nagar, Bengaluru, Karnataka 560078, India",
            structured["TRANSPORT SUMMARY - PART 1"]["Starting Location"],
        )
        self.assertIn(
            "Karnataka 560008, India",
            structured["TRANSPORT SUMMARY - PART 1"]["#1 Destination Address"],
        )

    def test_columnar_below_line_values_stay_with_their_keys(self):
        words = [
            {"text": "First", "bbox": [10, 10, 40, 20], "label": "B-KEY"},
            {"text": "Name:", "bbox": [44, 10, 80, 20], "label": "B-KEY"},
            {"text": "Last", "bbox": [130, 10, 160, 20], "label": "B-KEY"},
            {"text": "Name:", "bbox": [164, 10, 200, 20], "label": "B-KEY"},
            {"text": "Barbara", "bbox": [12, 32, 56, 42], "label": "B-VALUE"},
            {"text": "Wilson", "bbox": [132, 32, 174, 42], "label": "B-VALUE"},
        ]

        structured = extract_structured(words)

        self.assertEqual(structured["GENERAL"]["First Name:"], "Barbara")
        self.assertEqual(structured["GENERAL"]["Last Name:"], "Wilson")

    def test_form32_pairs_values_above_parenthetical_keys(self):
        with open(ROOT / "debug" / "model" / "Referral-form-template-32-Year.docx_1.json") as handle:
            labeled_words = json.load(handle)

        words = [
            {
                "text": item["words"],
                "bbox": ast.literal_eval(item["bbox"]),
                "label": item["label"],
            }
            for item in labeled_words
        ]

        structured = extract_structured(words)

        self.assertEqual(structured["GENERAL"]["DATE OF REFERRAL:"], "08/11/2025")
        self.assertEqual(structured["GENERAL"]["PATIENT SURNAME, FIRST"], "Thompson, Emily")
        self.assertEqual(structured["GENERAL"]["PREVIOUS / MAIDEN NAME"], "Roberts")
        self.assertEqual(structured["GENERAL"]["DOB: YY/MM/DD"], "90/05/15")
        self.assertEqual(structured["GENERAL"]["AGE"], "35")
        self.assertEqual(structured["GENERAL"]["PHN"], "(604) 458-5696")
        self.assertEqual(structured["GENERAL"]["ADDRESS"], "123 Maple Street. Vancouver, BC, V5K 1A2")
        self.assertEqual(structured["GENERAL"]["HOME PHONE"], "(604) 287-2658")
        self.assertEqual(structured["GENERAL"]["WORK PHONE"], "(589) 268-5987")
        self.assertEqual(structured["GENERAL"]["CELL PHONE"], "(685) 256-5487")


if __name__ == "__main__":
    unittest.main()
