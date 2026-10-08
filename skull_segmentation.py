"""Independent nnU-Net training/prediction workflow; see README.md.

CLI: prepare, train, predict. Inside Slicer: exec this file to define
export_training_case() and load_prediction(); no training runs automatically.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from uuid import uuid4

DATASET_NAME = "Dataset501_Skull"


def grouped_splits(cases):
    groups = sorted({case["animal_id"] for case in cases})
    if len(groups) < 2:
        raise ValueError("At least two independent animals are needed for held-out validation.")
    return [{"train": sorted(c["id"] for c in cases if c["animal_id"] != group),
             "val": sorted(c["id"] for c in cases if c["animal_id"] == group)}
            for group in groups]


def environment(workspace):
    root = Path(workspace).resolve()
    env = os.environ.copy()
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        directory = root / key
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    return env


def run_entry(command, arguments, env=None):
    env = (os.environ if env is None else env).copy()
    env["nnUNet_extTrainer"] = str(Path(__file__).resolve().parent / "model" / "nnunet_trainers")
    env["nnUNet_compile"] = "false"
    # Windows workers each import PyTorch/CUDA, exhausting commit memory on
    # smaller machines. Zero uses nnU-Net's single-process augmentation path.
    if os.name == "nt":
        if command == "nnUNetv2_train":
            env.setdefault("nnUNet_n_proc_DA", "0")
        else:
            # The planner also uses this setting for PyTorch CPU threads;
            # unlike the training augmenter, it cannot accept zero.
            env["nnUNet_n_proc_DA"] = str(max(1, int(env.get("nnUNet_n_proc_DA", "1"))))
    # Resolve the official installed console entry point in the chosen Python.
    code = ("import sys; from importlib.metadata import distribution; "
            "command_name=sys.argv.pop(1); "
            "entry=next(e for e in distribution('nnunetv2').entry_points "
            "if e.group=='console_scripts' and e.name==command_name); "
            "sys.exit(entry.load()())")
    subprocess.run([sys.executable, "-c", code, command, *map(str, arguments)],
                   env=env, check=True)


def prepare(manifest_path, workspace, allow_single_volume=False):
    import numpy as np
    import SimpleITK as sitk
    manifest_path = Path(manifest_path).resolve()
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))["cases"]
    if not cases or (len(cases) == 1 and not allow_single_volume):
        raise ValueError("Supply at least two labelled volumes, or use --allow-single-volume for a learning experiment.")
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9-]+", s) for s in ids):
        raise ValueError("Case IDs must be unique and contain letters, numbers or hyphens.")
    if any(not isinstance(c["animal_id"], str) or not c["animal_id"].strip() for c in cases):
        raise ValueError("Each case needs an animal_id for leakage-free splitting.")
    splits = [] if len(cases) == 1 else grouped_splits(cases)
    if not splits:
        print("Single-volume experiment: train fold 'all'. No independent validation is available.")
    destination = Path(environment(workspace)["nnUNet_raw"]) / DATASET_NAME
    if destination.exists():
        raise FileExistsError(f"Dataset already exists; use a new workspace: {destination}")
    # Validate all pairs before writing a dataset. Reject geometry mismatches;
    # aligned geometry alone does not establish anatomical registration.
    validated = []
    for case in cases:
        image_path = manifest_path.parent / case["image"]
        label_path = manifest_path.parent / case["label"]
        if str(label_path).lower().endswith(".seg.nrrd"):
            raise ValueError("Export just the skull as a binary labelmap, not a .seg.nrrd.")
        image, label = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(label_path))
        if image.GetDimension() != 3 or label.GetDimension() != 3:
            raise ValueError(f"{case['id']}: expected 3D image and label.")
        if image.GetNumberOfComponentsPerPixel() != 1 or label.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"{case['id']}: expected scalar data.")
        for getter in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
            if not np.allclose(getattr(image, getter)(), getattr(label, getter)(), rtol=0, atol=1e-5):
                raise ValueError(f"{case['id']}: image/label geometry differs ({getter}).")
        values = sitk.GetArrayFromImage(label)
        if not np.isin(values, [0, 1]).all() or not (values == 1).any():
            raise ValueError(f"{case['id']}: labels must contain background=0 and skull=1 only.")
        if not np.isfinite(sitk.GetArrayFromImage(image)).all():
            raise ValueError(f"{case['id']}: MRI contains nonfinite values.")
        validated.append((case["id"], image, sitk.Cast(label, sitk.sitkUInt8)))
    (destination / "imagesTr").mkdir(parents=True)
    (destination / "labelsTr").mkdir()
    for case_id, image, label in validated:
        sitk.WriteImage(image, str(destination / "imagesTr" / f"{case_id}_0000.nii.gz"))
        sitk.WriteImage(label, str(destination / "labelsTr" / f"{case_id}.nii.gz"))
    metadata = {"channel_names": {"0": "MRI"}, "labels": {"background": 0, "skull": 1},
                "numTraining": len(cases), "file_ending": ".nii.gz"}
    (destination / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (destination / "splits_final.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Prepared {len(cases)} volumes; {len(splits)} leave-one-animal-out folds in {destination}")


def preprocessing_complete(root, raw_dataset, configuration):
    dataset = Path(root) / DATASET_NAME
    plans_file = dataset / "nnUNetPlans.json"
    if not plans_file.is_file():
        return False
    plans = json.loads(plans_file.read_text(encoding="utf-8"))
    config = plans.get("configurations", {}).get(configuration)
    if not config or not (dataset / "dataset.json").is_file():
        return False
    data = dataset / config["data_identifier"]
    cases = [p.name.removesuffix(".nii.gz") for p in (raw_dataset / "labelsTr").glob("*.nii.gz")]
    def present(path):
        return path.is_file() and path.stat().st_size > 0
    return bool(cases) and all(
        present(data / f"{case}.pkl") and (
            (present(data / f"{case}.b2nd") and present(data / f"{case}_seg.b2nd"))
            or present(data / f"{case}.npz")) for case in cases)


def ensure_preprocessing(workspace, configuration, env, initial_model=None):
    workspace = Path(workspace).resolve()
    raw_dataset = Path(env["nnUNet_raw"]) / DATASET_NAME
    marker = workspace / f"preprocessing_{configuration}.json"
    root = Path(env["nnUNet_preprocessed"])
    if marker.is_file():
        root = Path(json.loads(marker.read_text(encoding="utf-8"))["root"])
        if not root.resolve().is_relative_to(workspace):
            raise ValueError("Preprocessing location must be within the model workspace.")
    if preprocessing_complete(root, raw_dataset, configuration):
        print(f"Reusing complete preprocessing: {root}", flush=True)
    else:
        # nnU-Net deletes its configuration folder before preprocessing. A
        # fresh root avoids locked/partially deleted Windows folders.
        # tempfile.mkdtemp on Windows/Python 3.11 creates an owner-only ACL.
        # Ordinary mkdir inherits project access for interactive user runs.
        root = workspace / f"preprocessed_{uuid4().hex[:8]}"
        root.mkdir()
        env["nnUNet_preprocessed"] = str(root)
        print(f"Preprocessing into a fresh directory: {root}", flush=True)
        if initial_model:
            dataset = root / DATASET_NAME
            dataset.mkdir()
            shutil.copyfile(Path(initial_model) / "plans.json", dataset / "nnUNetPlans.json")
            shutil.copyfile(raw_dataset / "dataset.json", dataset / "dataset.json")
            run_entry("nnUNetv2_extract_fingerprint", ["-d", "501", "-np", "1",
                      "--verify_dataset_integrity"], env)
            run_entry("nnUNetv2_preprocess", ["-d", "501", "-c", configuration, "-np", "1"], env)
        else:
            run_entry("nnUNetv2_plan_and_preprocess", ["-d", "501", "-c", configuration,
                      "--verify_dataset_integrity", "-np", "1", "-npfp", "1"], env)
        if not preprocessing_complete(root, raw_dataset, configuration):
            raise RuntimeError("Preprocessing finished without complete data for the requested configuration.")
    env["nnUNet_preprocessed"] = str(root)
    # Older initial-model runs produced cached patches but omitted this
    # metadata. Repair those caches without planning or preprocessing again.
    fingerprint = root / DATASET_NAME / "dataset_fingerprint.json"
    if not fingerprint.is_file():
        print("Generating missing dataset fingerprint.", flush=True)
        run_entry("nnUNetv2_extract_fingerprint", ["-d", "501", "-np", "1",
                  "--verify_dataset_integrity"], env)
        if not fingerprint.is_file():
            raise RuntimeError("Fingerprint extraction did not produce dataset_fingerprint.json.")
    marker.write_text(json.dumps({"root": str(root.resolve())}, indent=2), encoding="utf-8")
    return root


def train(workspace, folds, configuration, device, trainer="nnUNetTrainerSkullFP32", preprocess_only=False, resume=False, initial_model=None, checkpoint="checkpoint_final.pth", epochs=None):
    if resume and initial_model:
        raise ValueError("Use --initial-model for a new dataset, or --resume for an existing run.")
    weights = Path(initial_model).resolve() / "fold_all" / checkpoint if initial_model else None
    if weights and not weights.is_file():
        raise FileNotFoundError(weights)
    env = environment(workspace)
    if weights:
        env["SKULL_INITIAL_WEIGHTS"] = str(weights)
    else:
        env.pop("SKULL_INITIAL_WEIGHTS", None)
    if epochs is not None:
        if epochs < 1:
            raise ValueError("Epoch count must be positive.")
        env["SKULL_NUM_EPOCHS"] = str(epochs)
    source = Path(env["nnUNet_raw"]) / DATASET_NAME / "splits_final.json"
    splits = json.loads(source.read_text(encoding="utf-8"))
    requested = folds or ([str(index) for index in range(len(splits))] if splits else ["all"])
    if any(f != "all" and (not f.isdigit() or int(f) >= len(splits)) for f in requested):
        raise ValueError("Unknown fold. Choose a held-out fold index or 'all'.")
    root = ensure_preprocessing(workspace, configuration, env, initial_model)
    target = root / DATASET_NAME / "splits_final.json"
    shutil.copyfile(source, target)
    if preprocess_only:
        print("Preprocessing ready. Training was not started.")
        return
    settings_file = Path(workspace).resolve() / f"training_settings_{trainer}_{configuration}.json"
    if resume:
        if not settings_file.is_file():
            raise FileNotFoundError("Resume requires training settings from a run created by this repository.")
        settings = json.loads(settings_file.read_text(encoding="utf-8"))
        if epochs is not None and epochs != settings["epochs"]:
            raise ValueError("Keep the original total epoch count when resuming.")
    else:
        settings = {"initial_lr": 0.001 if weights else 0.01, "epochs": epochs or 1000}
        settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    env["SKULL_INITIAL_LR"] = str(settings["initial_lr"])
    env["SKULL_NUM_EPOCHS"] = str(settings["epochs"])
    for fold in requested:
        if resume:
            folder = Path(env["nnUNet_results"]) / DATASET_NAME / f"{trainer}__nnUNetPlans__{configuration}" / f"fold_{fold}"
            if not any((folder / name).is_file() for name in (
                    "checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth")):
                raise FileNotFoundError(f"No training checkpoint to resume in {folder}; start a new run first.")
        run_entry("nnUNetv2_train", ["501", configuration, fold, "-device", device,
                                   "-tr", trainer] + (["--c"] if resume else []), env)


def predict(image_path, model_folder, output_folder, folds, device, checkpoint="checkpoint_final.pth"):
    import tempfile
    import SimpleITK as sitk
    output = Path(output_folder).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "skull.nii.gz").exists():
        raise FileExistsError("Prediction exists; select a different output folder.")
    with tempfile.TemporaryDirectory(prefix="skull-predict-") as directory:
        image = sitk.ReadImage(str(Path(image_path).resolve()))
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("Prediction input must be a scalar 3D MRI.")
        sitk.WriteImage(image, str(Path(directory) / "skull_0000.nii.gz"))
        run_entry("nnUNetv2_predict_from_modelfolder",
                  ["-i", directory, "-o", output, "-m", Path(model_folder).resolve(),
                   "-f", *folds, "-chk", checkpoint, "-device", device, "-npp", "1", "-nps", "1"])
    result = output / "skull.nii.gz"
    if not result.exists():
        raise RuntimeError("nnU-Net did not produce the expected prediction.")
    print(f"Prediction: {result}")
    return result


def export_training_case(volume, segmentation, segment_id, directory, case_id):
    """Run in Slicer; export exactly one handmade segment in volume geometry."""
    import numpy as np
    import slicer
    if not segmentation.IsA("vtkMRMLSegmentationNode"):
        raise ValueError("Select a segmentation node; import the STL model into a segmentation first.")
    segments = segmentation.GetSegmentation()
    if not segment_id or segments.GetSegment(segment_id) is None:
        names = [segments.GetSegment(segments.GetNthSegmentID(i)).GetName()
                 for i in range(segments.GetNumberOfSegments())]
        raise ValueError(f"Choose a valid segment ID. Available segment names: {names}")
    if volume.GetParentTransformNode() or segmentation.GetParentTransformNode():
        raise ValueError("Harden image/segmentation transforms before exporting.")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    image_path, label_path = directory / f"{case_id}.nii.gz", directory / f"{case_id}_label.nii.gz"
    if image_path.exists() or label_path.exists():
        raise FileExistsError("Export paths already exist.")
    values = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segment_id, volume)
    label = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    try:
        label.CopyOrientation(volume)
        slicer.util.updateVolumeFromArray(label, (values > 0).astype(np.uint8))
        if not slicer.util.saveNode(volume, str(image_path)) or not slicer.util.saveNode(label, str(label_path)):
            raise RuntimeError("Training case export failed.")
    finally:
        slicer.mrmlScene.RemoveNode(label)
    return str(image_path), str(label_path)


def load_prediction(path, volume):
    """Run in Slicer; import model output as a separate editable segmentation."""
    import slicer
    label = slicer.util.loadLabelVolume(str(path))
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ModelSkullPrediction")
    try:
        node.CreateDefaultDisplayNodes()
        if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label, node):
            raise RuntimeError("Could not import predicted skull.")
        node.SetReferenceImageGeometryParameterFromVolumeNode(volume)
        node.CreateClosedSurfaceRepresentation()
        slicer.util.setSliceViewerLayers(background=volume)
    except Exception:
        slicer.mrmlScene.RemoveNode(node)
        raise
    finally:
        slicer.mrmlScene.RemoveNode(label)
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--workspace", required=True)
    prepare_parser.add_argument("--allow-single-volume", action="store_true",
                                help="Allow a one-volume experiment with no held-out validation.")
    training = commands.add_parser("train")
    training.add_argument("--workspace", required=True)
    training.add_argument("--folds", nargs="+")
    training.add_argument("--configuration", choices=["2d", "3d_fullres"], default="3d_fullres")
    training.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    training.add_argument("--trainer", default="nnUNetTrainerSkullFP32",
                          help="Default disables unstable FP16 and uses smaller 2D patches/batches.")
    training.add_argument("--preprocess-only", action="store_true",
                          help="Prepare/reuse preprocessing without starting training.")
    training.add_argument("--resume", action="store_true",
                          help="Continue from a saved checkpoint using nnU-Net --c.")
    training.add_argument("--initial-model", help="Model folder whose plans and fold_all weights initialize a new dataset run.")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--image", required=True)
    prediction.add_argument("--model-folder", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--folds", nargs="+", default=["0", "1", "2"])
    prediction.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    prediction.add_argument("--checkpoint", default="checkpoint_final.pth",
                            choices=["checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"])
    training.add_argument("--checkpoint", default="checkpoint_final.pth", choices=["checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"])
    training.add_argument("--epochs", type=int, help="Total epoch count; keep unchanged when resuming.")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.workspace, args.allow_single_volume)
    elif args.command == "train":
        train(args.workspace, args.folds, args.configuration, args.device, args.trainer, args.preprocess_only, args.resume, args.initial_model, args.checkpoint, args.epochs)
    else:
        predict(args.image, args.model_folder, args.output, args.folds, args.device, args.checkpoint)


if __name__ == "__main__":
    try:
        import slicer
        inside_slicer = hasattr(slicer, "mrmlScene")
    except ImportError:
        inside_slicer = False
    if inside_slicer:
        print("Model helpers loaded: export_training_case() and load_prediction(). See README.md.")
    else:
        main()
