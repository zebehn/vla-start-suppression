"""The kit's name is defined here and nowhere else.

Renaming: change NAME here, the [project] name in libero_goal/pyproject.toml, and the package directory name.
Code inside the package uses relative imports only, and tests/test_name.py fails if the literal name appears elsewhere.
"""
NAME = "wnta"
SPEC_VERSION = "v0"
SPEC_ID = f"{NAME}-{SPEC_VERSION}"
