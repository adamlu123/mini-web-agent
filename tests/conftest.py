import sys
from pathlib import Path

from cuawright.environments.osworld.source import activate_source


activate_source()

REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
if REPOSITORY_ROOT in sys.path:
    sys.path.remove(REPOSITORY_ROOT)
sys.path.insert(0, REPOSITORY_ROOT)
