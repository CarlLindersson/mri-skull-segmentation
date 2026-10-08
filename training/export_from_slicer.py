"""Load with exec(open(...).read()) in Slicer to define export_training_case.
Set SKULL_REPOSITORY to the folder where you downloaded this repository.
"""
from pathlib import Path
import sys
if "SKULL_REPOSITORY" not in globals():
    raise ValueError("Set SKULL_REPOSITORY to your repository folder first.")
sys.path.insert(0, str(Path(SKULL_REPOSITORY).resolve()))
from skull_segmentation import export_training_case
