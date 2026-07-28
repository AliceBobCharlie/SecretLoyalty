import sys
from pathlib import Path

# Repo root on sys.path so `import slaudit` works without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
