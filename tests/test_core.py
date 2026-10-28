import unittest

from private_ai_box import GPUS, MODELS, bom, configure, decode_tps, vram_needed
from private_ai_box.core import kv_cache_gb


class Tests(unittest.TestCase):
    def test_kv_cache_70b(self):
        # 2 * 80 * 8 * 128 * 2 bytes = 327,680 bytes/token -> 8k ctx ~= 2.68 GB
        self.assertAlmostEqual(kv_cache_gb(MODELS["70b"], 8192, 1), 2.684, places=2)

    def test_vram_quant_ordering(self):
        m = MODELS["70b"]
        self.assertLess(vram_needed(m, "int4", 4096, 1), vram_needed(m, "fp16", 4096, 1))
        self.assertGreater(vram_needed(m, "fp16", 4096, 1), 140)

    def test_moe_faster_than_dense(self):
        g = GPUS[-1]
        self.assertGreater(decode_tps(MODELS["moe-120b"], "int4", g, 1), decode_tps(MODELS["70b"], "int4", g, 1))

    def test_configure_sorted_and_fits(self):
        opts = configure("32b", "int4", users=20, concurrency=0.2, context=8192, tps_per_user=10)
        self.assertTrue(opts)
        self.assertEqual(opts, sorted(opts, key=lambda o: o["gpu_cost"]))
        for o in opts:
            self.assertGreaterEqual(o["vram_total"], o["vram_needed"])
            self.assertGreaterEqual(o["tps_per_user"], 10)

    def test_impossible_and_bom(self):
        self.assertEqual(configure("70b", "fp16", 500, 1.0, 128000, 50), [])
        opts = configure("8b", "int8", 5, 0.2, 4096, 20)
        b = bom(opts[0], ["hipaa"])
        self.assertGreater(b["suggested_price"], b["total"])
        self.assertIn("hipaa", b["compliance"])

    def test_margin_controls_suggested_price(self):
        opts = configure("8b", "int8", 5, 0.2, 4096, 20)
        b = bom(opts[0], margin=1.5)
        self.assertEqual(b["suggested_price"], round(b["total"] * 1.5, -2))
        with self.assertRaises(ValueError):
            bom(opts[0], margin=0)
        with self.assertRaises(ValueError):
            bom(opts[0], margin=float("nan"))

    def test_inputs_that_would_undersize_the_box_are_rejected(self):
        for args in (
            ("8b", "int8", 5, 0.2, 0, 20),  # context 0 drops the KV cache
            ("8b", "int8", 5, 3.0, 4096, 20),  # 300% of users generating at once
            ("8b", "int8", 0, 0.2, 4096, 20),
            ("8b", "int8", 5, 0.2, 4096, float("nan")),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                configure(*args)


if __name__ == "__main__":
    unittest.main()
