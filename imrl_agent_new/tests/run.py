"""
Run the entire test suite with
    python -m tests          # equivalent to: python tests/run.py
or
    pytest                   # if you prefer the vanilla CLI
"""
import sys
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # ensure project root is importable

if __name__ == "__main__":
    # -q  : quiet, only failing tests shown
    # -s  : show print output (useful for debugging)
    sys.exit(pytest.main(["-q", str(ROOT / "tests")]))
