# MRI skull segmentation

MRI skull segmentation using nnU-Net, pretrained on BlackBone MRI sequences, with model downloads, prediction tools, and fine-tuning support.

## 1. Set up the environment

Use Anaconda Prompt or a conda-enabled PowerShell terminal. Install Miniconda or Anaconda first if `conda` is unavailable.

Download the code using **Code > Download ZIP** on [GitHub](https://github.com/CarlLindersson/mri-skull-segmentation) and extract it, or clone it:

```powershell
git clone https://github.com/CarlLindersson/mri-skull-segmentation.git
cd mri-skull-segmentation
```

If you downloaded a ZIP, open a terminal in the extracted repository folder instead. Then create and install the environment:

```powershell
conda create -n skull-segmentation python=3.11 -y
conda activate skull-segmentation
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

These commands use an NVIDIA GPU with a compatible driver. Run prediction in this terminal, outside Slicer's Python Console. On later runs, open the repository folder and activate the environment with `conda activate skull-segmentation`.

## 2. Download the model

Open [GitHub Releases](https://github.com/CarlLindersson/mri-skull-segmentation/releases) and download the model ZIP from the release assets. The source-code ZIP does not contain the model weights. If no model release is available, obtain an exported model package from the maintainer.

Extract the model contents into `model/` so the files are arranged like this:

```text
model/
  plans.json
  dataset.json
  nnunet_trainers/nnUNetTrainerSkullFP32.py
  fold_all/checkpoint_final.pth
```

Check that `plans.json` is directly inside `model/`, rather than inside another nested folder. Some packages contain `checkpoint_best.pth` instead of `checkpoint_final.pth`; use that filename in the next command.

## 3. Run model prediction

With the environment active and the terminal in the repository folder, run:

```powershell
python inference/predict.py --image "C:\data\your_mri.nii.gz" --model-folder model --output "C:\data\skull-prediction001" --folds all --checkpoint checkpoint_final.pth
```

Replace the MRI path with your scan and the output path with a new folder. Both `.nii` and `.nii.gz` inputs work. Keep quotation marks around paths containing spaces. If your model contains `checkpoint_best.pth`, replace the checkpoint filename in the command.

The result is saved as:

```text
C:\data\skull-prediction001\skull.nii.gz
```

For another scan, use a different output folder. If CUDA is unavailable, install a suitable CPU PyTorch build and add `--device cpu` to the prediction command.

## 4. Inspect the result

In **3D Slicer**:

1. Load the original MRI.
2. Add the predicted `skull.nii.gz`. In the loading dialog, enable **Show Options** and select **LabelMap**.
3. Open **Segmentations** and create a new segmentation.
4. Under **Export/import models and labelmaps**, choose **Import**, select the predicted labelmap, and import it.
5. Open **Segment Editor**, select the new segmentation, and set the original MRI as the source volume.
6. Inspect the overlay in all three slice views. Enable **Show 3D** to inspect the skull surface and adjust colour/opacity in the segmentation display settings.
7. Correct errors if needed and use **Save** to save the segmentation.

Check for missed skull, gaps, and unwanted skin, especially around the eyes and frontal skull. The prediction uses the original MRI's geometry. It does not automatically remove islands, fill holes, or smooth the final skull.

## Fine-tuning

To prepare your own labels, fine-tune the model, and resume training, follow the [fine-tuning guide](training/README.md). Keep actual scans, labels, and training workspaces outside this repository.

## Export a release

Export from your trusted local training run:

```powershell
python export_model.py --source "C:\data\skull-training\nnUNet_results\Dataset501_Skull\nnUNetTrainerSkullFP32__nnUNetPlans__2d" --output "C:\data\model-release"
```

Use `--checkpoint checkpoint_best.pth` for best weights. The exporter supports this FP32 2D, single-MRI-channel, two-class, z-score-normalized model. It retains only inference weights and required metadata; removes optimizer state, logging, private initialization settings and unnecessary training statistics; and copies the required trainer. Only load trusted PyTorch checkpoints.

Test the export before publishing. ZIP the exported contents so they extract directly into `model/`, attach the ZIP to a versioned GitHub Release, and add download/checksum/evaluation notes here. Checkpoints are ignored by ordinary Git. Choose a license before public distribution. Training scans and actual manifests are never needed in the release.

## Test

```powershell
python tests/smoke_test.py --model-folder model
```

Add `--checkpoint checkpoint_best.pth` if needed. The CUDA test uses synthetic MRI/labels, prepares and preprocesses them, verifies complete weight preservation, runs one training and validation batch, restores a checkpoint, and predicts using the sanitized release. It checks binary output and original geometry. Outputs stay in ignored `.test-output/`. This checks functionality, not segmentation accuracy. Slicer export needs a separate visual check inside Slicer.

All processing runs locally; cloud synchronization depends on your storage settings.

## Citation

Please cite the following paper when using nnU-Net:

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods, 18*(2), 203–211.

If you use the tools or model distributed by this repository, please also cite the repository:

CarlLindersson. (2026). *MRI skull segmentation* [Computer software]. GitHub. https://github.com/CarlLindersson/mri-skull-segmentation

BibTeX reference handle: `CarlLindersson2026MRISkullSegmentation`.

```bibtex
@misc{CarlLindersson2026MRISkullSegmentation,
  author       = {{CarlLindersson}},
  title        = {{MRI} skull segmentation},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/CarlLindersson/mri-skull-segmentation},
  note         = {Computer software}
}
```

For reproducibility, include the release tag or commit hash you used. This is a software repository citation, not a publication citation. GitHub can generate repository references using the included `CITATION.cff` file.
