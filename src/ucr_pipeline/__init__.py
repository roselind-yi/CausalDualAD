# UCR Pipeline — 修复后的干净 UCR 流水线

from .model import DualPathModel, UCRDataset, create_dataloader, create_windows_vectorized
from .trainer import UCRTrainer
from .data import load_ucr_dataset, list_available_ucr, preprocess_ucr, analyze_dataset_characteristics
from .evaluation import UCREvaluator
from .config import default_config, load_config, save_config, merge_config

__all__ = [
    "DualPathModel",
    "UCRDataset",
    "create_dataloader",
    "create_windows_vectorized",
    "UCRTrainer",
    "load_ucr_dataset",
    "list_available_ucr",
    "preprocess_ucr",
    "analyze_dataset_characteristics",
    "UCREvaluator",
    "default_config",
    "load_config",
    "save_config",
    "merge_config",
]
