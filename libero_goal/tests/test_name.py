import os
import re
import unittest

from tests._util import BENCH

from wnta import _name


class NameDefinedOnce(unittest.TestCase):
    def test_package_code_has_no_literal_name(self):
        pkg = os.path.join(BENCH, _name.NAME)
        hits = []
        for root, _, files in os.walk(pkg):
            for f in files:
                if f.endswith((".py", ".json", ".yaml", ".md")) and f != "_name.py":
                    p = os.path.join(root, f)
                    with open(p, errors="replace") as fh:
                        if re.search(rf"\b{_name.NAME}\b", fh.read()):
                            hits.append(os.path.relpath(p, BENCH))
        self.assertEqual(hits, [], "use relative imports and _name.NAME instead of the literal package name")


if __name__ == "__main__":
    unittest.main()
