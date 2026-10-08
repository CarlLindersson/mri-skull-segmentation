# Segment a skull from an MRI

Follow these four steps to predict a skull segmentation and inspect it in Slicer.

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

Download the [model weights (v0.1.0)](https://github.com/CarlLindersson/mri-skull-segmentation/releases/download/v0.1.0/skull-model-v0.1.0.zip.zip) from the [published release](https://github.com/CarlLindersson/mri-skull-segmentation/releases/tag/v0.1.0). The source-code ZIP does not contain the model weights. The download currently has a `.zip.zip` filename; extract it as a normal ZIP archive.

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
