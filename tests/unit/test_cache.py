from __future__ import annotations

import unittest
from dataclasses import replace

from vision_jev.runtime import CacheIdentity


class CacheIdentityTest(unittest.TestCase):
    def test_every_revision_changes_key(self) -> None:
        base = CacheIdentity("i", "t", "p", "pc", "m", "a", "belief", "crop")
        self.assertNotEqual(base.key(), replace(base, adapter_revision="a2").key())
        self.assertNotEqual(base.key(), replace(base, mode="policy").key())
        self.assertNotEqual(base.key(), replace(base, crop_or_augmentation_sha256="crop2").key())


if __name__ == "__main__":
    unittest.main()
