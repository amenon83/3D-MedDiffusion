"""
Validate that the LungSBRT NIfTI volume loads correctly through the
3D-MedDiffusion dataloader (VQGANDataset / torchio) and has the right
shape, range, and dtype for the PatchVolume autoencoder.

Also demonstrates how a TOPAS dose distribution (numpy array) would be
converted to the same format.

Usage:
    python scripts/validate_input.py
"""

import os
import sys
import json
import numpy as np
import nibabel as nib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, "data", "LungSBRT")
CONFIG_JSON = os.path.join(REPO_ROOT, "config", "PatchVolume_data_LungSBRT.json")


def check_nifti_file():
    """Validate the NIfTI file on disk."""
    nii_path = os.path.join(DATA_DIR, "LungSBRTPatient.nii.gz")
    assert os.path.isfile(nii_path), f"NIfTI not found at {nii_path}"

    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine

    print("=== NIfTI file check ===")
    print(f"  Path   : {nii_path}")
    print(f"  Shape  : {data.shape}")
    print(f"  Dtype  : {data.dtype}")
    print(f"  Range  : [{data.min():.4f}, {data.max():.4f}]")
    print(f"  Affine diagonal (spacing): {np.diag(affine)[:3]}")
    print(f"  Affine origin            : {affine[:3, 3]}")

    assert data.ndim == 3, f"Expected 3D, got {data.ndim}D"
    assert 0.0 <= data.min() and data.max() <= 1.0 + 1e-6, \
        f"Values should be in [0,1] for 3DMedDiffusion; got [{data.min()}, {data.max()}]"
    print("  [PASS] NIfTI file is valid\n")
    return data, affine


def check_torchio_load():
    """Validate torchio can load and patch-sample the volume."""
    import torchio as tio

    nii_path = os.path.join(DATA_DIR, "LungSBRTPatient.nii.gz")
    scalar_img = tio.ScalarImage(nii_path)

    print("=== torchio ScalarImage check ===")
    print(f"  Shape (C,D,H,W) : {scalar_img.shape}")
    print(f"  Spacing          : {scalar_img.spacing}")
    data_tensor = scalar_img.data
    print(f"  Tensor dtype     : {data_tensor.dtype}")
    print(f"  Tensor range     : [{data_tensor.min():.4f}, {data_tensor.max():.4f}]")

    patch_size = 64
    sampler = tio.data.UniformSampler(patch_size)
    subject = tio.Subject(image=scalar_img)
    patch = next(sampler(subject))["image"]
    print(f"  Patch shape      : {patch.shape}  (should be [1, {patch_size}, {patch_size}, {patch_size}])")
    assert patch.shape == (1, patch_size, patch_size, patch_size), \
        f"Unexpected patch shape: {patch.shape}"
    print("  [PASS] torchio load + patch sampling works\n")
    return patch


def check_dataloader_pipeline():
    """Simulate what VQGANDataset.__getitem__ does."""
    import torch
    import torchio as tio

    nii_path = os.path.join(DATA_DIR, "LungSBRTPatient.nii.gz")
    whole_img = tio.ScalarImage(nii_path)

    patch_size = 64
    sampler = tio.data.UniformSampler(patch_size)
    img = next(sampler(tio.Subject(image=whole_img)))["image"]

    imageout = img.data                           # [1, D, H, W]
    imageout = imageout * 2 - 1                   # [0,1] -> [-1,1]
    imageout = imageout.transpose(1, 3).transpose(2, 3)  # [1, W, D, H] (matches VQGAN convention)
    imageout = imageout.type(torch.float32)

    print("=== Dataloader output simulation ===")
    print(f"  Output shape : {imageout.shape}")
    print(f"  Output dtype : {imageout.dtype}")
    print(f"  Output range : [{imageout.min():.4f}, {imageout.max():.4f}]")
    assert imageout.shape == (1, patch_size, patch_size, patch_size)
    assert -1.0 - 1e-6 <= imageout.min() and imageout.max() <= 1.0 + 1e-6, \
        f"Expected [-1,1] range, got [{imageout.min()}, {imageout.max()}]"
    print("  [PASS] Dataloader output matches expected format\n")
    return imageout


def check_config_json():
    """Verify the JSON config points to the data directory."""
    print("=== Config JSON check ===")
    with open(CONFIG_JSON) as f:
        cfg = json.load(f)
    print(f"  Config: {CONFIG_JSON}")
    for key, path in cfg.items():
        exists = os.path.isdir(path)
        nii_count = len([f for f in os.listdir(path) if f.endswith(".nii.gz")]) if exists else 0
        status = "OK" if exists and nii_count > 0 else "MISSING"
        print(f"  [{status}] {key} -> {path}  ({nii_count} .nii.gz files)")
    print()


def demo_dose_to_nifti():
    """
    Demonstrate how a TOPAS dose distribution (3D numpy array) would be
    converted to the same NIfTI format the diffusion model expects.
    """
    print("=== Demo: TOPAS dose array -> NIfTI ===")

    dose_shape = (128, 128, 128)
    dose = np.random.exponential(0.5, size=dose_shape).astype(np.float32)
    dose = np.clip(dose, 0, 5.0)
    print(f"  Simulated dose shape: {dose.shape}, range [{dose.min():.3f}, {dose.max():.3f}] Gy")

    d_min, d_max = 0.0, dose.max()
    dose_norm = (dose - d_min) / (d_max - d_min + 1e-8)
    dose_norm = dose_norm.astype(np.float32)

    spacing = [2.0, 2.0, 2.0]
    origin = [0.0, 0.0, 0.0]
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = origin

    nii = nib.Nifti1Image(dose_norm, affine)
    demo_path = os.path.join(DATA_DIR, "demo_dose.nii.gz")
    nib.save(nii, demo_path)
    print(f"  Saved demo dose NIfTI: {demo_path}")
    print(f"  Normalized range: [{dose_norm.min():.4f}, {dose_norm.max():.4f}]")

    reload = nib.load(demo_path).get_fdata(dtype=np.float32)
    assert reload.shape == dose_shape
    assert 0.0 <= reload.min() and reload.max() <= 1.0 + 1e-6
    print("  [PASS] Dose NIfTI round-trips correctly\n")

    os.remove(demo_path)


if __name__ == "__main__":
    print(f"Repo root: {REPO_ROOT}\n")
    check_nifti_file()
    check_config_json()
    check_torchio_load()
    check_dataloader_pipeline()
    demo_dose_to_nifti()
    print("All checks passed.")
