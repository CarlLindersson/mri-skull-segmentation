# MRI skull segmentation

MRI skull segmentation using nnU-Net, pretrained on BlackBone MRI sequences, with model downloads, prediction tools, and fine-tuning support.

## 1. Set up the environment

Use Anaconda Prompt or a conda-enabled PowerShell terminal. Install Miniconda or Anaconda first if `conda` is unavailable.

Download the code using **Code > Download ZIP** on [GitHub](https://github.com/CarlLindersson/mri-skull-segmentation) and extract it, or clone it:

```powershell
git clone https://github.com/CarlLindersson/mri-skull-segmentation.git
cd mri-skull-segmentation
```

If you downloaded a ZIP, open a terminal in the extracted repository folder instead. **Prediction works with or without an NVIDIA GPU.** Choose one setup below.

**With an NVIDIA GPU** (compatible driver required):

```powershell
conda create -n skull-segmentation python=3.11 -y
conda activate skull-segmentation
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

**Without an NVIDIA GPU: use the CPU** (no CUDA installation required):

```powershell
conda create -n skull-segmentation-cpu python=3.11 -y
conda activate skull-segmentation-cpu
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Both setups use the same model download. Run prediction in this terminal, outside Slicer's Python Console. On later runs, activate the environment you created: `conda activate skull-segmentation` for GPU, or `conda activate skull-segmentation-cpu` for CPU.

## 2. Download the model

Download the [model weights (v0.1.0)](https://github.com/CarlLindersson/mri-skull-segmentation/releases/download/v0.1.0/skull-model-v0.1.0.zip.zip) from this link or from the [published release](https://github.com/CarlLindersson/mri-skull-segmentation/releases/tag/v0.1.0). The source-code ZIP does not contain the model weights.

Extract the model contents into `model/` so the files are arranged like this:

```text
model/
  plans.json
  dataset.json
  nnunet_trainers/nnUNetTrainerSkullFP32.py
  fold_all/checkpoint_best.pth
```

Check that `plans.json` is directly inside `model/`, rather than inside another nested folder. This release contains `checkpoint_best.pth`; the prediction command below uses that checkpoint.

## 3. Run model prediction

With the environment active and the terminal in the repository folder, choose the matching command.

**NVIDIA GPU:**

```powershell
python inference/predict.py --image "C:\data\your_mri.nii.gz" --model-folder model --output "C:\data\skull-prediction001" --folds all --checkpoint checkpoint_best.pth
```

**CPU (no NVIDIA GPU needed):**

```powershell
python inference/predict.py --image "C:\data\your_mri.nii.gz" --model-folder model --output "C:\data\skull-prediction001" --folds all --checkpoint checkpoint_best.pth --device cpu
```

CPU prediction generally takes longer. The repository integration tests used CUDA; CPU prediction has not yet been verified end-to-end.

Replace the MRI path with your scan and the output path with a new folder. Both `.nii` and `.nii.gz` inputs work. Keep quotation marks around paths containing spaces. If you use a different release, match the checkpoint filename to that package.

The result is saved as:

```text
C:\data\skull-prediction001\skull.nii.gz
```

For another scan, use a different output folder. The output location and Slicer inspection steps are the same for GPU and CPU prediction.

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
