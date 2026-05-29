"""
Dataset loader for BrainMetShare MRI data
Supports multi-modal input and efficient 3D patch-based loading for consumer GPUs
"""

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional


class BrainMetDataset(Dataset):
    """
    Dataset for brain metastasis MRI segmentation

    Args:
        data_dir: Path to train or test directory
        sequences: List of MRI sequences to load (default: all 4)
        patch_size: Size of 3D patches for training (None = full volume)
        transform: Optional transforms
        metadata_path: Path to metadata CSV file
    """

    def __init__(
        self,
        data_dir: str,
        sequences: List[str] = ['t1_pre', 't1_gd', 'flair', 'bravo'],
        patch_size: Optional[Tuple[int, int, int]] = None,
        transform=None,
        metadata_path: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.sequences = sequences
        self.patch_size = patch_size
        self.transform = transform

        # Get all case directories
        self.cases = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('Mets_')])

        # Load metadata if provided
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
            self.metadata['Patient ID'] = self.metadata['Patient ID'].astype(str)

        # Check if this is training data (has segmentation masks)
        self.has_masks = (self.data_dir.name == 'train')

        print(f"Loaded {len(self.cases)} cases from {data_dir}")
        print(f"Sequences: {sequences}")
        print(f"Has segmentation masks: {self.has_masks}")

    def __len__(self):
        return len(self.cases)

    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load NIfTI file and return numpy array"""
        nii = nib.load(str(path))
        return nii.get_fdata().astype(np.float32)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Z-score normalization per volume"""
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / std
        return img

    def __getitem__(self, idx: int):
        case_dir = self.cases[idx]
        case_id = case_dir.name

        # Load all sequences
        images = []
        for seq in self.sequences:
            img_path = case_dir / f"{seq}.nii.gz"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing sequence {seq} for case {case_id}")

            img = self._load_nifti(img_path)
            img = self._normalize(img)
            images.append(img)

        # Stack sequences along channel dimension -> (C, H, W, D)
        images = np.stack(images, axis=0)

        # Load segmentation mask if available
        mask = None
        if self.has_masks:
            mask_path = case_dir / "seg.nii.gz"
            if mask_path.exists():
                mask = self._load_nifti(mask_path)
                # Convert to binary (0 or 1)
                mask = (mask > 0).astype(np.float32)
                mask = np.expand_dims(mask, axis=0)  # Add channel dim

        # Apply transforms if provided
        if self.transform:
            sample = {'image': images, 'mask': mask, 'case_id': case_id}
            sample = self.transform(sample)
            images = sample['image']
            mask = sample['mask']

        # Extract patch if specified (for memory efficiency)
        if self.patch_size is not None and mask is not None:
            images, mask = self._extract_random_patch(images, mask)

        # Convert to torch tensors
        images = torch.from_numpy(images).float()

        if mask is not None:
            mask = torch.from_numpy(mask).float()
            return images, mask, case_id
        else:
            return images, case_id

    def _extract_random_patch(
        self,
        images: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract random 3D patch from volume
        Biased towards patches containing metastases
        Ensures all patches are exactly patch_size dimensions
        """
        C, H, W, D = images.shape
        ph, pw, pd = self.patch_size

        # Ensure volume is large enough for patch extraction
        if H < ph or W < pw or D < pd:
            # Pad if volume is smaller than patch size
            pad_h = max(0, ph - H)
            pad_w = max(0, pw - W)
            pad_d = max(0, pd - D)

            images = np.pad(images, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            mask = np.pad(mask, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            C, H, W, D = images.shape

        # Find foreground voxels (where mask > 0)
        foreground = np.where(mask[0] > 0)

        # 80% chance to sample from foreground, 20% random
        if len(foreground[0]) > 0 and np.random.rand() > 0.2:
            # Sample center from foreground
            idx = np.random.randint(len(foreground[0]))
            ch = foreground[0][idx]
            cw = foreground[1][idx]
            cd = foreground[2][idx]
        else:
            # Random center
            ch = np.random.randint(ph//2, H - ph//2)
            cw = np.random.randint(pw//2, W - pw//2)
            cd = np.random.randint(pd//2, D - pd//2)

        # Calculate start coordinates centered on chosen point
        h_start = ch - ph//2
        w_start = cw - pw//2
        d_start = cd - pd//2

        # Ensure patch stays within bounds by adjusting start coordinates
        h_start = max(0, min(h_start, H - ph))
        w_start = max(0, min(w_start, W - pw))
        d_start = max(0, min(d_start, D - pd))

        # Extract patch with exact dimensions
        h_end = h_start + ph
        w_end = w_start + pw
        d_end = d_start + pd

        img_patch = images[:, h_start:h_end, w_start:w_end, d_start:d_end]
        mask_patch = mask[:, h_start:h_end, w_start:w_end, d_start:d_end]

        # Verify patch size (should always be true now)
        assert img_patch.shape == (C, ph, pw, pd), f"Patch shape mismatch: {img_patch.shape} vs expected {(C, ph, pw, pd)}"
        assert mask_patch.shape == (1, ph, pw, pd), f"Mask shape mismatch: {mask_patch.shape} vs expected {(1, ph, pw, pd)}"

        return img_patch, mask_patch

    def get_metadata(self, case_id: str):
        """Get metadata for a specific case"""
        if self.metadata is None:
            return None

        # Extract patient ID from case_id (e.g., "Mets_040" -> "40")
        patient_id = case_id.split('_')[1].lstrip('0') or '0'

        row = self.metadata[self.metadata['Patient ID'] == patient_id]
        if len(row) > 0:
            return row.iloc[0].to_dict()
        return None


def get_train_val_split(data_dir: str, val_ratio: float = 0.15, seed: int = 42):
    """
    Create train/validation split from training directory

    Args:
        data_dir: Path to training directory
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility

    Returns:
        train_cases, val_cases: Lists of case directory paths
    """
    import random
    random.seed(seed)

    data_dir = Path(data_dir)
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('Mets_')])

    # Shuffle and split
    cases_shuffled = cases.copy()
    random.shuffle(cases_shuffled)

    val_size = int(len(cases) * val_ratio)
    val_cases = cases_shuffled[:val_size]
    train_cases = cases_shuffled[val_size:]

    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")

    return train_cases, val_cases


if __name__ == "__main__":
    # Test dataset loading
    print("Testing BrainMetDataset...")

    dataset = BrainMetDataset(
        data_dir="../train",
        patch_size=(96, 96, 96),
        metadata_path="../metadata.csv"
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load first sample
    images, mask, case_id = dataset[0]
    print(f"\nFirst sample: {case_id}")
    print(f"Image shape: {images.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Mask unique values: {torch.unique(mask)}")

    # Get metadata
    metadata = dataset.get_metadata(case_id)
    if metadata:
        print(f"Metadata: {metadata}")
