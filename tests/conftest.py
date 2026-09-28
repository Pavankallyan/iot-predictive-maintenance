import os
import sys

# Let tests import the repo's src/ modules as top-level packages,
# mirroring how src/infer.py imports `preprocess`.
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
