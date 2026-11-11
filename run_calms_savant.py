"""Source-tree launcher for machines with NumPy already installed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from calms_savant.cli import main
if __name__ == "__main__":
    main()
