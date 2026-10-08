# Segment a skull from an MRI

Follow these four steps to predict a skull segmentation and inspect it in Slicer.

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
