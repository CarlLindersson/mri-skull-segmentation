"""Local integration test using synthetic images and a trusted model.

Run: python tests/smoke_test.py --model-folder MODEL --checkpoint checkpoint_best.pth
No private scans are read or copied. Outputs remain in an ignored test directory.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skull_segmentation import environment, prepare, train, predict
from export_model import export_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-folder', required=True)
    parser.add_argument('--checkpoint', default='checkpoint_final.pth')
    args = parser.parse_args()
    test_root = ROOT / '.test-output' / uuid4().hex[:8]
    test_root.mkdir(parents=True)
    model = export_model(args.model_folder, test_root / 'model', args.checkpoint)
    y, x = np.mgrid[:64, :64]
    radius = np.sqrt((x - 32)**2 + (y - 32)**2)
    skull = ((radius > 20) & (radius < 23)).astype(np.uint8)
    image = np.repeat((50 * (radius < 24) + 2 * skull)[None].astype(np.float32), 3, axis=0)
    labels = np.repeat(skull[None], 3, axis=0)
    for name, values in [('case001', image), ('case001_label', labels)]:
        node = sitk.GetImageFromArray(values)
        node.SetSpacing((0.5, 0.5, 0.5))
        sitk.WriteImage(node, str(test_root / f'{name}.nii.gz'))
    manifest = {'cases': [{'id': 'case001', 'animal_id': 'animal001', 'image': 'case001.nii.gz', 'label': 'case001_label.nii.gz'}]}
    manifest_path = test_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    workspace = test_root / 'workspace'
    prepare(manifest_path, workspace, allow_single_volume=True)
    train(workspace, ['all'], '2d', 'cuda', preprocess_only=True, initial_model=model, checkpoint=args.checkpoint)
    env = environment(workspace)
    env['nnUNet_preprocessed'] = json.loads((workspace / 'preprocessing_2d.json').read_text())['root']
    os.environ.update(env)
    os.environ['nnUNet_compile'] = 'false'
    os.environ['nnUNet_n_proc_DA'] = '0'
    os.environ['SKULL_INITIAL_WEIGHTS'] = str(model / 'fold_all' / args.checkpoint)
    sys.path.insert(0, str(ROOT / 'model/nnunet_trainers'))
    import torch
    from nnUNetTrainerSkullFP32 import nnUNetTrainerSkullFP32
    dataset = Path(env['nnUNet_preprocessed']) / 'Dataset501_Skull'
    plans = json.loads((dataset / 'nnUNetPlans.json').read_text())
    metadata = json.loads((dataset / 'dataset.json').read_text())
    plans['continue_training'] = False
    trainer = nnUNetTrainerSkullFP32(plans, '2d', 'all', metadata)
    trainer.num_epochs = 1
    trainer.num_iterations_per_epoch = 1
    trainer.num_val_iterations_per_epoch = 1
    trainer.initialize()
    checkpoint = torch.load(model / 'fold_all' / args.checkpoint, map_location='cpu', weights_only=False)
    assert set(checkpoint) == {'network_weights', 'trainer_name', 'init_args', 'inference_allowed_mirroring_axes'}
    assert checkpoint['init_args'] == {'configuration': '2d'}
    for key, value in trainer.network.state_dict().items():
        assert torch.equal(value.cpu(), checkpoint['network_weights'][key]), key
    print('PASS: every weight, including output layers, preserved.', flush=True)
    del checkpoint
    trainer.run_training()
    assert all(torch.isfinite(p).all() for p in trainer.network.parameters())
    trained_model = Path(trainer.output_folder).parent
    final = torch.load(Path(trainer.output_folder) / 'checkpoint_final.pth', map_location='cpu', weights_only=False)
    assert final['current_epoch'] == 1
    assert final['optimizer_state']['state']
    del final
    trainer.load_checkpoint(str(Path(trainer.output_folder) / 'checkpoint_final.pth'))
    assert trainer.current_epoch == 1
    import skull_segmentation as pipeline
    actual_entry = pipeline.run_entry
    captured = []
    pipeline.run_entry = lambda command, arguments, env=None: captured.append(env.copy())
    try:
        pipeline.train(workspace, ['all'], '2d', 'cuda', initial_model=model, checkpoint=args.checkpoint, epochs=1)
        pipeline.train(workspace, ['all'], '2d', 'cuda', resume=True)
        assert captured[-1]['SKULL_INITIAL_LR'] == '0.001'
        assert captured[-1]['SKULL_NUM_EPOCHS'] == '1'
    finally:
        pipeline.run_entry = actual_entry
    print('PASS: resume retains fine-tuning learning rate and total epochs.', flush=True)
    print('PASS: training/validation epoch and checkpoint restoration.', flush=True)
    del trainer
    torch.cuda.empty_cache()
    os.environ.pop('SKULL_INITIAL_WEIGHTS', None)
    # Verify the sanitized release itself, rather than the full training checkpoint.
    prediction = predict(test_root / 'case001.nii.gz', model, test_root / 'prediction', ['all'], 'cuda', args.checkpoint)
    result = sitk.ReadImage(str(prediction))
    original = sitk.ReadImage(str(test_root / 'case001.nii.gz'))
    for getter in ['GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection']:
        assert np.allclose(getattr(result, getter)(), getattr(original, getter)())
    assert np.isin(sitk.GetArrayFromImage(result), [0, 1]).all()
    print('PASS: sanitized model prediction, binary labels, and original geometry.', flush=True)
    print(f'All integration checks passed. Local outputs: {test_root}', flush=True)


if __name__ == '__main__':
    main()
