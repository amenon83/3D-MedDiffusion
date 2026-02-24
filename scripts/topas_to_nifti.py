"""
Bridge: convert SIEMAC/TOPAS dose, dose-rate, and LET arrays to NIfTI
volumes that 3D-MedDiffusion can consume.

The SIEMAC pipeline stores per-spot dose/LET in pickle archives
(D0_Ggji / L0_Ggji). Before optimization, the optimizer sums them into
full 3D arrays (d_i, DR_i, LET_i). This script shows how to take those
aggregated 3D numpy arrays and write them as NIfTI files compatible with
the 3D-MedDiffusion dataloader.

Two modes:
  1. "from_pickles" - load D0_Ggji.pickle.bz2 / L0_Ggji.pickle.bz2
     and constants.pickle, aggregate d_i / DR_i / LET_i, then save.
  2. "from_arrays" - accept pre-built numpy arrays directly (for use
     inside the optimization loop).

Usage (standalone):
    python scripts/topas_to_nifti.py \
        --pickles-dir /path/to/outputs/pickles \
        --output-dir  data/LungSBRT_dose \
        --channels dose dr let

Usage (as library inside the optimization loop):
    from topas_to_nifti import arrays_to_nifti, normalize_volume
    ...
    dose_nii = arrays_to_nifti(dose_3d, constants, channel="dose")
"""

import argparse
import os
import sys
import bz2
import pickle

import numpy as np
import nibabel as nib


def build_affine(constants: dict) -> np.ndarray:
    """Build a NIfTI affine from the SIEMAC cropped-CT grid parameters."""
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = constants["cctdx"]
    affine[1, 1] = constants["cctdy"]
    affine[2, 2] = constants["cctdz"]
    affine[0, 3] = constants["cctxmin"]
    affine[1, 3] = constants["cctymin"]
    affine[2, 3] = constants["cctzmin"]
    return affine


def normalize_volume(vol: np.ndarray, clip_lo: float = 0.0,
                     clip_hi: float | None = None) -> np.ndarray:
    """Normalize a 3D array to [0, 1] for 3D-MedDiffusion.

    The dataloader later maps [0,1] -> [-1,1] via  img*2-1.
    """
    if clip_hi is None:
        clip_hi = float(np.percentile(vol[vol > 0], 99.5)) if (vol > 0).any() else 1.0
    vol = np.clip(vol, clip_lo, clip_hi)
    denom = clip_hi - clip_lo
    if denom < 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - clip_lo) / denom).astype(np.float32)


def arrays_to_nifti(vol_3d: np.ndarray, constants: dict,
                    channel: str = "dose",
                    output_path: str | None = None,
                    clip_hi: float | None = None) -> nib.Nifti1Image:
    """Convert a single 3D numpy array (e.g. dose, DR, or LET) to a
    [0,1]-normalized NIfTI image aligned with the cropped CT grid.

    Parameters
    ----------
    vol_3d : np.ndarray of shape (cctnx, cctny, cctnz)
    constants : dict from constants.pickle
    channel : str, label for the output file name
    output_path : optional path to save the .nii.gz
    clip_hi : upper clipping value before normalization (None = auto 99.5th pct)

    Returns
    -------
    nib.Nifti1Image (values in [0, 1])
    """
    assert vol_3d.ndim == 3, f"Expected 3D array, got shape {vol_3d.shape}"
    norm = normalize_volume(vol_3d, clip_hi=clip_hi)
    affine = build_affine(constants)
    nii = nib.Nifti1Image(norm, affine)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        nib.save(nii, output_path)
        print(f"Saved {channel} NIfTI: {output_path}  shape={norm.shape}  "
              f"range=[{norm.min():.4f}, {norm.max():.4f}]")
    return nii


def aggregate_dose_dr_let(D0_Ggji, L0_Ggji, spotws, constants):
    """Reproduce the optimizer's dose/DR/LET aggregation.

    Parameters
    ----------
    D0_Ggji : list[list[list[ndarray]]]  — D0_Ggji[field][geom][spot]
    L0_Ggji : list[list[list[ndarray]]]  — same structure for LET
    spotws  : list[np.ndarray]           — spot weights per field
    constants : dict

    Returns
    -------
    (dose_3d, dr_3d, let_3d) each of shape (cctnx, cctny, cctnz)
    """
    nx = constants["cctnx"]
    ny = constants["cctny"]
    nz = constants["cctnz"]
    wu = constants["weightunit"]
    I = constants["beamcurrent"]
    qe = 1.6e-19

    dose_3d = np.zeros((nx, ny, nz), dtype=np.float64)
    dr_num  = np.zeros((nx, ny, nz), dtype=np.float64)
    let_num = np.zeros((nx, ny, nz), dtype=np.float64)

    geom_idx = 0  # default geometry
    for iG in range(len(D0_Ggji)):
        ws = spotws[iG]
        for j in range(len(D0_Ggji[iG][geom_idx])):
            Dij = D0_Ggji[iG][geom_idx][j]
            Lij = L0_Ggji[iG][geom_idx][j]
            if not isinstance(Dij, np.ndarray):
                continue
            wj = ws[j]
            dij = wj * Dij
            tj = qe * wu * wj / I if wj > 0 else 1.0
            dose_3d += dij
            dr_num  += dij * dij / tj
            let_num += dij * Lij

    safe_dose = np.where(dose_3d > 0, dose_3d, 1.0)
    dr_3d  = dr_num  / safe_dose
    let_3d = let_num / safe_dose
    dr_3d[dose_3d <= 0]  = 0.0
    let_3d[dose_3d <= 0] = 0.0

    return dose_3d.astype(np.float32), dr_3d.astype(np.float32), let_3d.astype(np.float32)


def from_pickles(pickles_dir: str, output_dir: str, channels: list[str]):
    """Load SIEMAC pickle outputs and write NIfTI volumes."""
    with open(os.path.join(pickles_dir, "constants.pickle"), "rb") as f:
        constants = pickle.load(f)

    with bz2.open(os.path.join(pickles_dir, "D0_Ggji.pickle.bz2"), "rb") as f:
        D0_Ggji = pickle.load(f)

    with bz2.open(os.path.join(pickles_dir, "L0_Ggji.pickle.bz2"), "rb") as f:
        L0_Ggji = pickle.load(f)

    nfields = len(D0_Ggji)
    nspots_per_field = [len(D0_Ggji[iG][0]) for iG in range(nfields)]
    spotws = [np.ones(n, dtype=np.float32) for n in nspots_per_field]
    print(f"Using uniform unit weights for {nfields} fields, "
          f"spots per field: {nspots_per_field}")

    dose_3d, dr_3d, let_3d = aggregate_dose_dr_let(
        D0_Ggji, L0_Ggji, spotws, constants
    )

    os.makedirs(output_dir, exist_ok=True)
    channel_map = {"dose": dose_3d, "dr": dr_3d, "let": let_3d}
    for ch in channels:
        arrays_to_nifti(
            channel_map[ch], constants, channel=ch,
            output_path=os.path.join(output_dir, f"{ch}.nii.gz"),
        )


def demo_synthetic(output_dir: str):
    """Create synthetic dose/DR/LET NIfTIs using the patient constants
    to validate the format without needing actual TOPAS outputs."""
    import importlib.util

    constants_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Accelerated_SIEMAC", "workflows", "patient2opt3f", "config", "constants_pat1.py",
    )
    if not os.path.isfile(constants_path):
        print(f"constants_pat1.py not found at {constants_path}, using defaults")
        constants = {
            "cctnx": 65, "cctny": 66, "cctnz": 22,
            "cctxmin": -54.6, "cctymin": -372.7, "cctzmin": -86.3,
            "cctdx": 3.906, "cctdy": 3.906, "cctdz": 4.5,
            "weightunit": 100000, "beamcurrent": 100.0e-9,
        }
    else:
        spec = importlib.util.spec_from_file_location("constants_pat1", constants_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["constants_pat1"] = mod
        # Patch pickle.dump to no-op so it doesn't try to write
        _orig_dump = pickle.dump
        pickle.dump = lambda *a, **k: None
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        finally:
            pickle.dump = _orig_dump
        constants = mod.constants

    nx, ny, nz = constants["cctnx"], constants["cctny"], constants["cctnz"]
    print(f"\nSynthetic demo on cropped CT grid: ({nx}, {ny}, {nz})")

    rng = np.random.default_rng(42)
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    xx, yy, zz = np.mgrid[:nx, :ny, :nz]
    r2 = ((xx - cx) / (nx * 0.15)) ** 2 + ((yy - cy) / (ny * 0.15)) ** 2 + ((zz - cz) / (nz * 0.15)) ** 2
    dose_3d = (2.0 * np.exp(-0.5 * r2) + 0.05 * rng.standard_normal((nx, ny, nz))).clip(0).astype(np.float32)
    dr_3d   = (40.0 * np.exp(-0.3 * r2) + rng.standard_normal((nx, ny, nz))).clip(0).astype(np.float32)
    let_3d  = (3.0 + 2.0 * r2 / r2.max() + 0.2 * rng.standard_normal((nx, ny, nz))).clip(0).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    for ch, arr in [("dose", dose_3d), ("dr", dr_3d), ("let", let_3d)]:
        arrays_to_nifti(arr, constants, channel=ch,
                        output_path=os.path.join(output_dir, f"{ch}.nii.gz"))

    print(f"\nSynthetic NIfTIs written to {output_dir}/")
    print("These can be loaded by 3D-MedDiffusion's VQGANDataset exactly "
          "like CT volumes.\n")


def main():
    p = argparse.ArgumentParser(
        description="Convert SIEMAC dose/DR/LET arrays to NIfTI for 3D-MedDiffusion"
    )
    sub = p.add_subparsers(dest="mode")

    pick = sub.add_parser("from-pickles",
                          help="Load D0/L0 pickles and aggregate")
    pick.add_argument("--pickles-dir", required=True)
    pick.add_argument("--output-dir", required=True)
    pick.add_argument("--channels", nargs="+", default=["dose", "dr", "let"],
                      choices=["dose", "dr", "let"])

    demo = sub.add_parser("demo",
                          help="Create synthetic NIfTIs for format validation")
    demo.add_argument("--output-dir", default="data/LungSBRT_dose")

    args = p.parse_args()
    if args.mode == "from-pickles":
        from_pickles(args.pickles_dir, args.output_dir, args.channels)
    elif args.mode == "demo":
        demo_synthetic(args.output_dir)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
