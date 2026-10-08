# Verification

Checked on 2026-10-08 with Python 3.11, nnunetv2 2.8.1, PyTorch 2.8.0+cu126, SimpleITK 2.5.6, and an NVIDIA GTX 1650 Ti.

The synthetic CUDA integration test passed:

- Exported a trusted checkpoint with only inference weights and metadata.
- Prepared a binary skull label and matching synthetic MRI.
- Verified dataset integrity and generated fingerprints/preprocessed caches.
- Compared every loaded weight against the source, including output layers.
- Completed one training batch and one validation batch with finite loss.
- Saved and restored the full training checkpoint and optimizer state.
- Verified resume configuration retains the fine-tuning learning rate and total epoch count.
- Ran prediction using the sanitized model package.
- Verified output labels are binary and match original size, spacing, origin, and direction.
- Checked prediction, preparation, and training CLI help and Python compilation.
- Separately verified fresh experiment planning and preprocessing without a pretrained model.

These are functionality checks, not evidence of segmentation accuracy. No private scans or handmade labels were used as test inputs. A local trained checkpoint was used to test model compatibility; neither it nor test outputs are included here. Slicer helpers have not been exercised in a live Slicer session as part of this verification.
