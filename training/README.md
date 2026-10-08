# Fine-tune on your own MRI volumes

Fine-tuning uses your own MRI volumes and aligned handmade skull segmentations. Keep all actual data and training outputs outside this repository.

## 1. Download and install

Download/clone the code from https://github.com/CarlLindersson/mri-skull-segmentation. From its directory:

```powershell
conda create -n skull-segmentation python=3.11 -y
conda activate skull-segmentation
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

The tested training setup uses a compatible NVIDIA GPU. An existing environment containing these dependencies also works.

## 2. Get initial weights

Download the model ZIP from the repository's **Releases** page, when available. Extract its contents into `model/`, so `plans.json`, `dataset.json`, and `fold_all/checkpoint_final.pth` are present. Code downloads alone do not include weights. If no release exists, export a trusted local model using the procedure in `inference/README.md`, then use its directory with `--initial-model` below.

## 3. Export aligned MRI/label pairs in Slicer

Load the MRI. Import the handmade skull STL into a segmentation using **Segmentations → Export/import models and labelmaps**. Check alignment in all slice views and harden transforms. Ensure the selected segment represents the skull consistently across cases.

In Slicer's Python Console:

```python
SKULL_REPOSITORY = r"C:\tools\mri-skull-segmentation"
exec(open(SKULL_REPOSITORY + r"\training\export_from_slicer.py").read())
volume = slicer.util.getNode("YOUR_MRI_NODE_NAME")
segmentation = slicer.util.getNode("YOUR_SKULL_SEGMENTATION_NODE_NAME")
segments = segmentation.GetSegmentation()
for i in range(segments.GetNumberOfSegments()):
    sid = segments.GetNthSegmentID(i)
    print(sid, segments.GetSegment(sid).GetName())
```

Choose the skull ID printed above, then export:

```python
segment_id = "YOUR_SKULL_SEGMENT_ID"
export_training_case(volume, segmentation, segment_id, r"C:\data\skull-labels", "case001")
```

Repeat with different case IDs. Each export creates an MRI and a binary labelmap on the same grid. Review exported labels visually. Matching geometry alone does not prove anatomical alignment.

## 4. Create your manifest

Copy `training/manifest.example.json` to `C:\data\skull-labels\manifest.json`. Replace generic entries with your own filenames and IDs. Paths resolve relative to the manifest. All scans from the same subject must use the same `animal_id` grouping field so held-out splits do not mix that subject across training and validation. This grouping field is user-supplied metadata, not a list of development subjects.

Labels must contain only background = 0 and skull = 1, with at least one skull voxel.

## 5. Prepare the dataset

Back in your terminal, from the repository directory:

```powershell
python training/prepare_dataset.py --manifest "C:\data\skull-labels\manifest.json" --workspace "C:\data\skull-training"
```

Use a new workspace when changing the dataset. A one-volume experiment needs `--allow-single-volume`, and must train with fold `all`.

## 6. Fine-tune

```powershell
python training/train.py --workspace "C:\data\skull-training" --configuration 2d --folds all --initial-model model --epochs 100
```

This retains all weights, including output layers; uses the released architecture/preprocessing; and starts a new optimizer at learning rate 0.001. All layers are trainable. The example schedules 100 epochs; the default is 1,000. For best weights, add `--checkpoint checkpoint_best.pth`. Change `--initial-model` if your local export is elsewhere.

Fold `all` trains and validates on the same supplied subjects. For independent validation, use a numeric held-out fold and ensure the initial model never trained on that subject. Preparation defines subject-level leave-one-out folds.

## 7. Stop and resume

Stop with Ctrl+C, preferably after an epoch finishes. Resume:

```powershell
python training/train.py --workspace "C:\data\skull-training" --configuration 2d --folds all --resume
```

The saved settings retain the original learning rate and total epoch schedule. Resume restores a full training checkpoint; progress since the last save may be lost. Do not combine `--resume` and `--initial-model`.

## 8. Predict with your fine-tuned model

```powershell
python inference/predict.py --image "C:\data\new_mri.nii.gz" --model-folder "C:\data\skull-training\nnUNet_results\Dataset501_Skull\nnUNetTrainerSkullFP32__nnUNetPlans__2d" --output "C:\data\fine-tuned-prediction001" --folds all --checkpoint checkpoint_best.pth
```

The output is `skull.nii.gz`. View it in Slicer using the prediction guide. Inspect full-volume anatomy; pseudo Dice on known subjects is not independent performance evidence.

## 9. Optional: train from scratch or share weights

To train from scratch, omit `--initial-model` from step 6. Scratch training starts at learning rate 0.01.

To share your model, use `export_model.py` as described in the root README. Never publish actual manifests, data, raw training checkpoints, or complete workspaces. The exported inference package removes training state and case metadata and can initialize another fine-tuning run.
