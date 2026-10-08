"""Export a local, trusted nnU-Net checkpoint as an inference-only package."""
import argparse
import json
from pathlib import Path
import shutil
import torch


def export_model(source, destination, checkpoint='checkpoint_final.pth'):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new output directory; existing packages are never overwritten.')
    plans = json.loads((source / 'plans.json').read_text(encoding='utf-8'))
    metadata = json.loads((source / 'dataset.json').read_text(encoding='utf-8'))
    if metadata['channel_names'] != {'0': 'MRI'} or metadata['labels'] != {'background': 0, 'skull': 1}:
        raise ValueError('This exporter supports one MRI channel and background/skull labels only.')
    saved = torch.load(source / 'fold_all' / checkpoint, map_location='cpu', weights_only=False)
    if saved['init_args']['configuration'] != '2d':
        raise ValueError('Expected the supported FP32 2D skull model.')
    allowed = ('plans_name', 'image_reader_writer', 'transpose_forward', 'transpose_backward')
    clean_plans = {key: plans[key] for key in allowed}
    clean_plans.update(dataset_name='Dataset501_Skull', configurations={'2d': plans['configurations']['2d']},
                       foreground_intensity_properties_per_channel={'0': {}})
    if clean_plans['configurations']['2d']['normalization_schemes'] != ['ZScoreNormalization']:
        raise ValueError('Only z-score normalization is supported by this metadata sanitization.')
    clean_checkpoint = {
        'network_weights': saved['network_weights'],
        'trainer_name': 'nnUNetTrainerSkullFP32',
        'init_args': {'configuration': '2d'},
        'inference_allowed_mirroring_axes': saved.get('inference_allowed_mirroring_axes'),
    }
    clean_metadata = {'channel_names': {'0': 'MRI'}, 'labels': {'background': 0, 'skull': 1}, 'file_ending': '.nii.gz'}
    (destination / 'fold_all').mkdir(parents=True)
    (destination / 'plans.json').write_text(json.dumps(clean_plans, indent=2), encoding='utf-8')
    (destination / 'dataset.json').write_text(json.dumps(clean_metadata, indent=2), encoding='utf-8')
    torch.save(clean_checkpoint, destination / 'fold_all' / checkpoint)
    shutil.copytree(Path(__file__).resolve().parent / 'model/nnunet_trainers', destination / 'nnunet_trainers',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    print(f'Inference-only model exported to {destination}')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', default='checkpoint_final.pth',
                        choices=['checkpoint_final.pth', 'checkpoint_best.pth', 'checkpoint_latest.pth'])
    args = parser.parse_args()
    export_model(args.source, args.output, args.checkpoint)
