import unittest

from sarcasm_detector import resolve_sentiment
from sarcasm_sentiment import extract_sarcasm_features


class SarcasmPolarityTests(unittest.TestCase):
    def test_mock_complaint_is_positive_sarcasm(self):
        text = (
            "Terrible product! Now my wife keeps stealing my charger because "
            "it charges in 10 minutes flat."
        )
        features = extract_sarcasm_features(text)

        self.assertGreater(features[0], 0)
        self.assertEqual(
            resolve_sentiment(
                "negative", {"sarcasm_type": "positive_sarcasm"}
            ),
            {"label": "positive", "adjusted": True},
        )

    def test_ironic_praise_after_failure_is_negative_sarcasm(self):
        text = (
            "Oh brilliant! The phone case broke on day one. "
            "Truly a masterpiece of engineering."
        )
        features = extract_sarcasm_features(text)

        self.assertGreater(features[1], 0)
        self.assertEqual(
            resolve_sentiment(
                "positive", {"sarcasm_type": "negative_sarcasm"}
            ),
            {"label": "negative", "adjusted": True},
        )

    def test_binary_sarcasm_detection_does_not_invert_sentiment(self):
        self.assertEqual(
            resolve_sentiment(
                "positive", {"sarcasm_type": "sarcasm_detected"}
            ),
            {"label": "positive", "adjusted": False},
        )


if __name__ == "__main__":
    unittest.main()
