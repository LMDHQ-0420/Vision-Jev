from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vision_jev.api_keys import load_api_keys


class APIKeysTest(unittest.TestCase):
    def test_loads_exact_secret_only_contract_without_repr_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api_keys.toml"
            path.write_text('[kimi]\napi_key="kimi-secret"\n[glm]\napi_key="glm-secret"\n')
            keys = load_api_keys(path)
            self.assertEqual(keys.kimi, "kimi-secret")
            self.assertEqual(keys.glm, "glm-secret")
            self.assertNotIn("secret", repr(keys))

    def test_rejects_extra_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api_keys.toml"
            path.write_text('[kimi]\napi_key="a"\nmodel="x"\n[glm]\napi_key="b"\n')
            with self.assertRaisesRegex(ValueError, "must contain only"):
                load_api_keys(path)

    def test_kimi_can_be_required_without_glm_balance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api_keys.toml"
            path.write_text('[kimi]\napi_key="kimi-secret"\n[glm]\napi_key=""\n')
            keys = load_api_keys(path, required_providers=("kimi",))
            self.assertEqual(keys.kimi, "kimi-secret")
            with self.assertRaisesRegex(ValueError, "glm"):
                load_api_keys(path)


if __name__ == "__main__":
    unittest.main()
