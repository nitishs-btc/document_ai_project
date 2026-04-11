import unittest

from src.checkbox import group_checkbox_results, map_checkbox_to_text


class CheckboxMappingTests(unittest.TestCase):
    def test_checkbox_results_include_question_context(self):
        words = [
            {"text": "Is", "bbox": [10, 10, 20, 20], "label": "O"},
            {"text": "this", "bbox": [24, 10, 40, 20], "label": "O"},
            {"text": "referral", "bbox": [44, 10, 78, 20], "label": "O"},
            {"text": "urgent?", "bbox": [82, 10, 124, 20], "label": "O"},
            {"text": "Yes", "bbox": [160, 10, 185, 20], "label": "B-OPTION"},
            {"text": "No", "bbox": [220, 10, 240, 20], "label": "B-OPTION"},
            {"text": "Check", "bbox": [10, 40, 35, 50], "label": "O"},
            {"text": "all", "bbox": [40, 40, 56, 50], "label": "O"},
            {"text": "that", "bbox": [60, 40, 82, 50], "label": "O"},
            {"text": "apply:", "bbox": [86, 40, 122, 50], "label": "O"},
            {"text": "Insurance", "bbox": [40, 70, 95, 80], "label": "B-OPTION"},
            {"text": "cards", "bbox": [100, 70, 132, 80], "label": "I-OPTION"},
        ]
        checkboxes = [
            {"bbox": [140, 8, 155, 23], "status": "unchecked"},
            {"bbox": [200, 8, 215, 23], "status": "checked"},
            {"bbox": [20, 68, 35, 83], "status": "checked"},
        ]

        mapped = map_checkbox_to_text(checkboxes, words)

        self.assertEqual(
            mapped,
            [
                {
                    "text": "Yes",
                    "option": "Yes",
                    "question": "Is this referral urgent?",
                    "section": "GENERAL",
                    "status": "unchecked",
                },
                {
                    "text": "No",
                    "option": "No",
                    "question": "Is this referral urgent?",
                    "section": "GENERAL",
                    "status": "checked",
                },
                {
                    "text": "Insurance cards",
                    "option": "Insurance cards",
                    "question": "Check all that apply:",
                    "section": "GENERAL",
                    "status": "checked",
                },
            ],
        )

    def test_checkbox_results_can_be_grouped_by_question(self):
        grouped = group_checkbox_results(
            [
                {
                    "text": "Yes",
                    "option": "Yes",
                    "question": "Is this referral urgent?",
                    "section": "REFERRAL TYPE",
                    "status": "unchecked",
                },
                {
                    "text": "No",
                    "option": "No",
                    "question": "Is this referral urgent?",
                    "section": "REFERRAL TYPE",
                    "status": "checked",
                },
            ]
        )

        self.assertEqual(
            grouped,
            {
                "REFERRAL TYPE": {
                    "Is this referral urgent?": {
                        "Yes": "unchecked",
                        "No": "checked",
                    }
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
