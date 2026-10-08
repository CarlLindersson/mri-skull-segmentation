"""CLI entry point; see README.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skull_segmentation import main

if __name__ == "__main__":
    sys.argv.insert(1, "prepare")
    main()
