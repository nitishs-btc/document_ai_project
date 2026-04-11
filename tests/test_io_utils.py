import unittest

from src.io_utils import group_image_paths


class GroupImagePathsTests(unittest.TestCase):
    def test_groups_multi_page_images_without_collapsing_singletons(self):
        grouped = group_image_paths(
            [
                "input/Referral-form-template-37.doc.docx_1.png",
                "input/Referral-form-template-37.doc.docx_2.png",
                "input/Referral-form-template-37.doc.docx_3.png",
                "input/Referral-form-template-37.doc.docx_4.png",
                "input/2026-597_1.png",
            ]
        )

        self.assertEqual(
            grouped,
            [
                {
                    "name": "2026-597_1",
                    "paths": ["input/2026-597_1.png"],
                    "multi_page": False,
                },
                {
                    "name": "Referral-form-template-37.doc.docx",
                    "paths": [
                        "input/Referral-form-template-37.doc.docx_1.png",
                        "input/Referral-form-template-37.doc.docx_2.png",
                        "input/Referral-form-template-37.doc.docx_3.png",
                        "input/Referral-form-template-37.doc.docx_4.png",
                    ],
                    "multi_page": True,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
