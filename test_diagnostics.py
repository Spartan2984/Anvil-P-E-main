#!/usr/bin/env python
"""Quick diagnostic to see what's happening."""
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except ImportError as e:
    print(f"NumPy import failed: {e}")

try:
    import scipy
    print(f"SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"SciPy import failed: {e}")

try:
    from data import make_patterns
    print("data module: OK")
except ImportError as e:
    print(f"data module failed: {e}")

print("Diagnostics complete.")
