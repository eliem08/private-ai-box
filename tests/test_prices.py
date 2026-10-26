import datetime
import json
import os
import tempfile
import unittest

from private_ai_box.core import GPUS, configure, load_prices, price_age_days

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "reseller-prices.json")


class PriceFileTests(unittest.TestCase):
    def test_override_and_new_gpu(self):
        gpus, as_of = load_prices(EXAMPLE)
        by = {g.name: g for g in gpus}
        self.assertEqual((as_of, by["H100 80GB"].price, by["H100 80GB"].vram_gb), ("2026-10-20", 26500, 80))
        self.assertEqual(by["B200 192GB"].vram_gb, 192)
        self.assertEqual(len(gpus), len(GPUS) + 1)
        opts = configure("70b", "int4", 200, 0.2, 8192, 15, True, gpus)
        self.assertIn("B200 192GB", {o["gpu"] for o in opts})

    def test_bad_files_are_rejected(self):
        bad = [
            {"gpus": [{"name": "H100 80GB", "price": 1}]},  # no date
            {"as_of": "2026-10-20", "gpus": [{"name": "H100 80GB", "price": -5}]},
            {"as_of": "2026-10-20", "gpus": [{"name": "Mystery GPU", "price": 900}]},  # new gpu without specs
            {"as_of": "soon", "gpus": []},
        ]
        with tempfile.TemporaryDirectory() as t:
            for i, doc in enumerate(bad):
                path = os.path.join(t, f"{i}.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(doc, fh)
                with self.subTest(doc=doc), self.assertRaises(ValueError):
                    load_prices(path)

    def test_price_age(self):
        self.assertEqual(price_age_days("2026-10-01", datetime.date(2026, 12, 30)), 90)


if __name__ == "__main__":
    unittest.main()
