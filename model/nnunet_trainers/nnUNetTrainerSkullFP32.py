"""FP32 nnU-Net for GPUs whose FP16 convolutions produce NaNs."""
import os
from copy import deepcopy
from functools import wraps
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def full_precision_forward(module):
    # Decorate forward rather than wrapping the module: checkpoint weight keys
    # remain identical, and reconstruction during prediction uses the same fix.
    original = module.forward
    @wraps(original)
    def forward(*args, **kwargs):
        with torch.autocast("cuda", enabled=False):
            return original(*args, **kwargs)
    module.forward = forward
    return module


class nnUNetTrainerSkullFP32(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        plans = deepcopy(plans)
        if configuration == "2d":
            config = plans["configurations"][configuration]
            config["batch_size"] = 2
            # Retain architecture-compatible divisibility and do not enlarge
            # smaller planned patches. This dataset's eight stages require 128.
            strides = np.prod(config["architecture"]["arch_kwargs"]["strides"], axis=0)
            config["patch_size"] = [min(int(size), max(int(divisor),
                                      (256 // int(divisor)) * int(divisor)))
                                    for size, divisor in zip(config["patch_size"], strides)]
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.grad_scaler = None
        if os.environ.get("SKULL_NUM_EPOCHS"):
            self.num_epochs = int(os.environ["SKULL_NUM_EPOCHS"])
        if os.environ.get("SKULL_INITIAL_WEIGHTS"):
            self.initial_lr = 0.001
        if os.environ.get("SKULL_INITIAL_LR"):
            self.initial_lr = float(os.environ["SKULL_INITIAL_LR"])

    def initialize(self):
        super().initialize()
        path = os.environ.get("SKULL_INITIAL_WEIGHTS")
        if path and not getattr(self, "_initial_weights_loaded", False):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            # Strict loading includes the skull/background output layers.
            self.network.load_state_dict(checkpoint["network_weights"], strict=True)
            self._initial_weights_loaded = True
            self.print_to_log_file("Loaded all model weights, including segmentation output layers.")

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        return full_precision_forward(nnUNetTrainer.build_network_architecture(
            plans_manager, configuration_manager, num_input_channels,
            num_output_channels, enable_deep_supervision))

    def _build_loss(self):
        return full_precision_forward(super()._build_loss())

    def train_step(self, batch):
        result = super().train_step(batch)
        if not np.isfinite(result["loss"]).all():
            raise FloatingPointError("Nonfinite FP32 training loss. Stop and check inputs/weights.")
        return result

    def validation_step(self, batch):
        result = super().validation_step(batch)
        if not np.isfinite(result["loss"]).all():
            raise FloatingPointError("Nonfinite FP32 validation loss. Stop and check inputs/weights.")
        return result
