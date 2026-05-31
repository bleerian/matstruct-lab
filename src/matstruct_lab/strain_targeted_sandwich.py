
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ase import Atoms
from ase.build import fcc111
from ase.io import read, write

from matstruct_lab.lattice_match import make_2d_supercell, inplane_cell
from matstruct_lab.heterostructures import set_inplane_cell, shift_z_min_to


DEFAULT_FCC_A = {
    "Au": 4.0782,
    "Ag": 4.0862,
    "Cu": 3.6149,
    "Pt": 3.9239,
    "Pd": 3.8907,
    "Ni": 3.5240,
}


DEFAULT_METAL_MATRIX = np.array([[-2,  2],
                                 [ 0, -2]], dtype=int)

DEFAULT_MOS2_MATRIX = np.array([[-1,  1],
                                [-1, -2]], dtype=int)


def safe_token(x: float | None) -> str:
    if x is None:
        return "none"
    return f"{float(x):.3f}".replace("-", "m").replace(".", "p")


def check_2d_cell(atoms: Atoms, name: str, tol: float = 1e-6) -> None:
    cell = atoms.cell.array
    if abs(cell[0, 2]) > tol or abs(cell[1, 2]) > tol:
        raise ValueError(
            f"{name}: first two lattice vectors have nonzero z components. "
            "Reorient so a,b are in-plane and c is out-of-plane."
        )


def find_structure_file(structure_dir: Path, name: str) -> Path | None:
    candidates = [
        structure_dir / f"{name}_111.cif",
        structure_dir / f"{name}_111.vasp",
        structure_dir / f"{name}_111.POSCAR",
        structure_dir / f"{name}111.cif",
        structure_dir / f"{name}111.vasp",
        structure_dir / f"{name}.cif",
        structure_dir / f"{name}.vasp",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_or_build_fcc111(
    metal: str,
    structure_dir: Path,
    fcc_a: dict[str, float] | None = None,
    layers: int = 4,
) -> tuple[Atoms, str]:
    path = find_structure_file(structure_dir, metal)
    if path is not None:
        atoms = read(path)
        atoms.set_pbc([True, True, False])
        check_2d_cell(atoms, metal)
        return atoms, f"file:{path}"

    fcc_a = DEFAULT_FCC_A if fcc_a is None else fcc_a
    if metal not in fcc_a:
        raise ValueError(f"No structure file or fcc lattice constant for {metal}")

    atoms = fcc111(
        metal,
        size=(1, 1, layers),
        a=fcc_a[metal],
        vacuum=10.0,
        orthogonal=False,
    )
    atoms.set_pbc([True, True, False])
    check_2d_cell(atoms, metal)
    return atoms, f"generated:fcc111(a={fcc_a[metal]}, layers={layers})"


def cell_metrics(cell2d: np.ndarray) -> dict[str, float]:
    cell2d = np.asarray(cell2d, dtype=float)
    a = cell2d[0]
    b = cell2d[1]

    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    if la == 0.0 or lb == 0.0:
        raise ValueError("Zero-length in-plane lattice vector.")

    cos_gamma = float(np.dot(a, b) / (la * lb))
    cos_gamma = float(np.clip(cos_gamma, -1.0, 1.0))

    return {
        "a_A": la,
        "b_A": lb,
        "gamma_deg": float(np.degrees(np.arccos(cos_gamma))),
        "area_A2": float(abs(np.linalg.det(cell2d))),
    }


def add_cell_record(prefix: str, cell2d: np.ndarray) -> dict[str, Any]:
    m = cell_metrics(cell2d)
    return {
        f"{prefix}_a_x": float(cell2d[0, 0]),
        f"{prefix}_a_y": float(cell2d[0, 1]),
        f"{prefix}_b_x": float(cell2d[1, 0]),
        f"{prefix}_b_y": float(cell2d[1, 1]),
        f"{prefix}_a_A": m["a_A"],
        f"{prefix}_b_A": m["b_A"],
        f"{prefix}_gamma_deg": m["gamma_deg"],
        f"{prefix}_area_A2": m["area_A2"],
    }


def strain_record(reference_cell: np.ndarray, target_cell: np.ndarray, prefix: str) -> dict[str, Any]:
    reference_cell = np.asarray(reference_cell, dtype=float)
    target_cell = np.asarray(target_cell, dtype=float)

    F = np.linalg.solve(reference_cell, target_cell)
    singular_values = np.linalg.svd(F, compute_uv=False)
    principal = singular_values - 1.0

    ref_m = cell_metrics(reference_cell)
    tar_m = cell_metrics(target_cell)
    signed_area_strain = (tar_m["area_A2"] - ref_m["area_A2"]) / ref_m["area_A2"]

    return {
        f"{prefix}_principal_strain_1_percent": 100.0 * float(principal[0]),
        f"{prefix}_principal_strain_2_percent": 100.0 * float(principal[1]),
        f"{prefix}_max_abs_principal_strain_percent": 100.0 * float(np.max(np.abs(principal))),
        f"{prefix}_rms_principal_strain_percent": 100.0 * float(np.sqrt(np.mean(principal**2))),
        f"{prefix}_signed_area_strain_percent": 100.0 * float(signed_area_strain),
        f"{prefix}_deformation_gradient": F.tolist(),
    }


def effective_scale(reference_cell: np.ndarray, target_cell: np.ndarray) -> float:
    """
    Geometric mean of the singular values mapping reference_cell -> target_cell.
    For nearly hexagonal/isotropic cases this is the effective linear scale.
    """
    F = np.linalg.solve(reference_cell, target_cell)
    svals = np.linalg.svd(F, compute_uv=False)
    return float(np.exp(np.mean(np.log(svals))))


def cell_with_reference_length_in_orientation(
    reference_cell: np.ndarray,
    orientation_cell: np.ndarray,
) -> np.ndarray:
    """
    Return a cell with the orientation/shear basis of orientation_cell but the
    effective length scale of reference_cell.
    """
    scale = effective_scale(orientation_cell, reference_cell)
    return np.asarray(orientation_cell, dtype=float) * scale


def alpha_target_cell(
    metal_cell: np.ndarray,
    mos2_cell: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Native alpha target in metal orientation.

    alpha=0.0 gives MoS2-like length in metal orientation.
    alpha=0.5 gives geometric shared strain.
    alpha=1.0 gives native metal cell.
    """
    metal_cell = np.asarray(metal_cell, dtype=float)
    mos2_cell = np.asarray(mos2_cell, dtype=float)

    scale_mos2_to_metal = effective_scale(mos2_cell, metal_cell)
    target = metal_cell * (scale_mos2_to_metal ** (alpha - 1.0))

    F = np.linalg.solve(mos2_cell, metal_cell)
    svals = np.linalg.svd(F, compute_uv=False)

    return target, {
        "target_policy": "native_alpha",
        "alpha": alpha,
        "native_scale_mos2_to_metal": scale_mos2_to_metal,
        "native_singular_value_1": float(svals[0]),
        "native_singular_value_2": float(svals[1]),
        "native_singular_value_spread": float(abs(svals[0] - svals[1])),
    }


def metal_strain_target_cell(
    metal_cell: np.ndarray,
    mos2_cell: np.ndarray,
    target_abs_metal_strain_percent: float = 4.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Target a fixed absolute metal strain, moving metal toward the MoS2 cell.

    This avoids forcing Ni/Cu/etc. to the literal Au in-plane cell.
    """
    scale_metal_to_mos2 = effective_scale(metal_cell, mos2_cell)

    # If MoS2-like cell is larger than metal, stretch metal; otherwise compress metal.
    sign = 1.0 if scale_metal_to_mos2 > 1.0 else -1.0
    signed_strain = sign * target_abs_metal_strain_percent / 100.0

    target = np.asarray(metal_cell, dtype=float) * (1.0 + signed_strain)

    return target, {
        "target_policy": "metal_abs_strain_target",
        "target_abs_metal_strain_percent": target_abs_metal_strain_percent,
        "signed_target_metal_strain_percent": 100.0 * signed_strain,
        "native_scale_metal_to_mos2": scale_metal_to_mos2,
    }


def mos2_strain_target_cell(
    metal_cell: np.ndarray,
    mos2_cell: np.ndarray,
    target_abs_mos2_strain_percent: float = 4.7,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Target a fixed absolute MoS2 strain, using metal orientation.

    This can still overstrain some metals. The generator can filter it out
    using max_allowed_other_strain_percent.
    """
    mos2_zero_in_metal_orientation = cell_with_reference_length_in_orientation(
        reference_cell=mos2_cell,
        orientation_cell=metal_cell,
    )

    scale_mos2_to_metal = effective_scale(mos2_cell, metal_cell)

    # If metal-like cell is larger than MoS2, stretch MoS2; otherwise compress MoS2.
    sign = 1.0 if scale_mos2_to_metal > 1.0 else -1.0
    signed_strain = sign * target_abs_mos2_strain_percent / 100.0

    target = mos2_zero_in_metal_orientation * (1.0 + signed_strain)

    return target, {
        "target_policy": "mos2_abs_strain_target",
        "target_abs_mos2_strain_percent": target_abs_mos2_strain_percent,
        "signed_target_mos2_strain_percent": 100.0 * signed_strain,
        "native_scale_mos2_to_metal": scale_mos2_to_metal,
    }


def build_sandwich_from_target_cell(
    metal_atoms: Atoms,
    mos2_atoms: Atoms,
    target_inplane: np.ndarray,
    metal_matrix: np.ndarray,
    mos2_matrix: np.ndarray,
    interlayer_distance: float,
    vacuum: float,
) -> Atoms:
    bottom_metal = make_2d_supercell(metal_atoms, metal_matrix)
    middle_sc = make_2d_supercell(mos2_atoms, mos2_matrix)
    top_metal = make_2d_supercell(metal_atoms, metal_matrix)

    bottom_metal = set_inplane_cell(bottom_metal, target_inplane, scale_atoms=True)
    middle_sc = set_inplane_cell(middle_sc, target_inplane, scale_atoms=True)
    top_metal = set_inplane_cell(top_metal, target_inplane, scale_atoms=True)

    bottom_metal = shift_z_min_to(bottom_metal, vacuum / 2.0)
    middle_sc = shift_z_min_to(
        middle_sc,
        bottom_metal.positions[:, 2].max() + interlayer_distance,
    )
    top_metal = shift_z_min_to(
        top_metal,
        middle_sc.positions[:, 2].max() + interlayer_distance,
    )

    stack = bottom_metal + middle_sc + top_metal

    z_min = stack.positions[:, 2].min()
    z_max = stack.positions[:, 2].max()
    final_z = (z_max - z_min) + vacuum

    final_cell = np.zeros((3, 3), dtype=float)
    final_cell[:2, :2] = target_inplane
    final_cell[2, 2] = final_z

    stack.set_cell(final_cell, scale_atoms=False)
    stack.set_pbc([True, True, False])
    stack.wrap(eps=1e-8)

    return stack


def write_all_formats(atoms: Atoms, outdir: Path, stem: str) -> dict[str, str]:
    paths = {
        "vasp": outdir / f"{stem}.vasp",
        "cif": outdir / f"{stem}.cif",
        "xyz": outdir / f"{stem}.xyz",
    }
    write(paths["vasp"], atoms, format="vasp", direct=True)
    write(paths["cif"], atoms)
    write(paths["xyz"], atoms, format="extxyz")
    return {k: str(v) for k, v in paths.items()}


def generate_strain_targeted_metal_mos2_sandwiches(
    structure_dir: str | Path = "../examples/structures",
    mos2_filename: str = "MoS2.cif",
    outdir: str | Path = "strain_targeted_metal_mos2_sandwiches",
    metals: list[str] | None = None,
    alphas: list[float] | None = None,
    distances: list[float] | np.ndarray | None = None,
    metal_matrix: np.ndarray = DEFAULT_METAL_MATRIX,
    mos2_matrix: np.ndarray = DEFAULT_MOS2_MATRIX,
    vacuum: float = 20.0,
    fcc111_layers: int = 4,
    target_abs_metal_strain_percent: float = 4.495,
    target_abs_mos2_strain_percent: float = 4.707,
    max_allowed_other_strain_percent: float = 6.0,
    include_native_alpha: bool = True,
    include_metal_target: bool = True,
    include_mos2_target: bool = True,
    write_structures: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate native-alpha and strain-targeted metal/MoS2/metal sandwiches.

    Returns
    -------
    summary_df
        One row per unique target cell.
    metadata_df
        One row per written structure.
    skipped_df
        Target cells skipped because the non-targeted material exceeded the cap.
    """
    structure_dir = Path(structure_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metals = ["Au", "Ag", "Cu", "Pt", "Pd", "Ni"] if metals is None else metals
    alphas = [0.0, 0.5, 1.0] if alphas is None else alphas
    distances = np.arange(2.5, 4.0001, 0.25) if distances is None else np.asarray(distances, dtype=float)

    mos2_path = structure_dir / mos2_filename
    if not mos2_path.exists():
        raise FileNotFoundError(f"Could not find MoS2 file: {mos2_path}")

    mos2 = read(mos2_path)
    mos2.set_pbc([True, True, False])
    check_2d_cell(mos2, "MoS2")

    mos2_sc = make_2d_supercell(mos2, mos2_matrix)
    mos2_ref_cell = inplane_cell(mos2_sc)

    metal_atoms_by_name: dict[str, Atoms] = {}
    metal_source_by_name: dict[str, str] = {}
    metal_ref_cell_by_name: dict[str, np.ndarray] = {}

    for metal in metals:
        atoms, source = load_or_build_fcc111(
            metal=metal,
            structure_dir=structure_dir,
            layers=fcc111_layers,
        )
        metal_atoms_by_name[metal] = atoms
        metal_source_by_name[metal] = source

        metal_sc = make_2d_supercell(atoms, metal_matrix)
        metal_ref_cell_by_name[metal] = inplane_cell(metal_sc)

        print(f"{metal}: {len(atoms)} atoms in input cell | {source}")

    print(f"MoS2: {len(mos2)} atoms in input cell | {mos2_path}")

    summary_records: list[dict[str, Any]] = []
    metadata_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []

    for metal in metals:
        metal_atoms = metal_atoms_by_name[metal]
        metal_ref_cell = metal_ref_cell_by_name[metal]

        jobs: list[dict[str, Any]] = []

        if include_native_alpha:
            for alpha in alphas:
                target, info = alpha_target_cell(
                    metal_cell=metal_ref_cell,
                    mos2_cell=mos2_ref_cell,
                    alpha=alpha,
                )
                if np.isclose(alpha, 0.0):
                    label = "native alpha=0; MoS2 fixed length, metal strained"
                elif np.isclose(alpha, 0.5):
                    label = "native alpha=0.5; balanced shared strain"
                elif np.isclose(alpha, 1.0):
                    label = "native alpha=1; metal fixed length, MoS2 strained"
                else:
                    label = f"native alpha={alpha}"

                jobs.append({
                    "series": "native_alpha",
                    "alpha": alpha,
                    "target_cell": target,
                    "target_info": info,
                    "mode_label": label,
                    "filter_other_material": False,
                })

        if include_metal_target:
            target, info = metal_strain_target_cell(
                metal_cell=metal_ref_cell,
                mos2_cell=mos2_ref_cell,
                target_abs_metal_strain_percent=target_abs_metal_strain_percent,
            )
            jobs.append({
                "series": f"metal_target_{safe_token(target_abs_metal_strain_percent)}pct",
                "alpha": None,
                "target_cell": target,
                "target_info": info,
                "mode_label": f"metal strain targeted to {target_abs_metal_strain_percent:.3f}%; MoS2 strain recorded",
                "filter_other_material": True,
            })

        if include_mos2_target:
            target, info = mos2_strain_target_cell(
                metal_cell=metal_ref_cell,
                mos2_cell=mos2_ref_cell,
                target_abs_mos2_strain_percent=target_abs_mos2_strain_percent,
            )
            jobs.append({
                "series": f"mos2_target_{safe_token(target_abs_mos2_strain_percent)}pct",
                "alpha": None,
                "target_cell": target,
                "target_info": info,
                "mode_label": f"MoS2 strain targeted to {target_abs_mos2_strain_percent:.3f}%; metal strain recorded",
                "filter_other_material": True,
            })

        for job in jobs:
            target_cell = job["target_cell"]

            metal_strain = strain_record(metal_ref_cell, target_cell, "metal")
            mos2_strain = strain_record(mos2_ref_cell, target_cell, "mos2")
            raw_mismatch = strain_record(mos2_ref_cell, metal_ref_cell, "raw_mos2_to_metal")

            base = {
                "series": job["series"],
                "metal": metal,
                "middle": "MoS2",
                "structure_type": f"{metal}/MoS2/{metal}",
                "alpha": job["alpha"],
                "mode_label": job["mode_label"],
                "metal_source": metal_source_by_name[metal],
                "mos2_source": str(mos2_path),
                "metal_matrix": metal_matrix.tolist(),
                "mos2_matrix": mos2_matrix.tolist(),
                "metal_input_atoms": len(metal_atoms),
                "mos2_input_atoms": len(mos2),
                "metal_supercell_atoms_one_electrode": len(metal_atoms) * abs(round(np.linalg.det(metal_matrix))),
                "mos2_supercell_atoms": len(mos2) * abs(round(np.linalg.det(mos2_matrix))),
                "sandwich_atoms_expected": (
                    2 * len(metal_atoms) * abs(round(np.linalg.det(metal_matrix)))
                    + len(mos2) * abs(round(np.linalg.det(mos2_matrix)))
                ),
                "max_allowed_other_strain_percent": max_allowed_other_strain_percent,
                **job["target_info"],
                **add_cell_record("metal_reference", metal_ref_cell),
                **add_cell_record("mos2_reference", mos2_ref_cell),
                **add_cell_record("target", target_cell),
                **raw_mismatch,
                **metal_strain,
                **mos2_strain,
            }

            metal_abs = base["metal_max_abs_principal_strain_percent"]
            mos2_abs = base["mos2_max_abs_principal_strain_percent"]

            skip = False
            skip_reason = ""

            if job["filter_other_material"]:
                if job["series"].startswith("metal_target") and mos2_abs > max_allowed_other_strain_percent:
                    skip = True
                    skip_reason = (
                        f"MoS2 residual strain {mos2_abs:.3f}% exceeds cap "
                        f"{max_allowed_other_strain_percent:.3f}%"
                    )

                if job["series"].startswith("mos2_target") and metal_abs > max_allowed_other_strain_percent:
                    skip = True
                    skip_reason = (
                        f"metal residual strain {metal_abs:.3f}% exceeds cap "
                        f"{max_allowed_other_strain_percent:.3f}%"
                    )

            summary_row = dict(base)
            summary_row["skipped"] = skip
            summary_row["skip_reason"] = skip_reason
            summary_records.append(summary_row)

            if skip:
                skipped_records.append(summary_row)
                print(
                    f"SKIP {metal} {job['series']} | "
                    f"metal={metal_abs:.3f}% | MoS2={mos2_abs:.3f}% | {skip_reason}"
                )
                continue

            for d in distances:
                atoms = build_sandwich_from_target_cell(
                    metal_atoms=metal_atoms,
                    mos2_atoms=mos2,
                    target_inplane=target_cell,
                    metal_matrix=metal_matrix,
                    mos2_matrix=mos2_matrix,
                    interlayer_distance=float(d),
                    vacuum=vacuum,
                )

                stem = (
                    f"{metal}_MoS2_{metal}"
                    f"_{job['series']}"
                    f"_alpha{safe_token(job['alpha'])}"
                    f"_d{safe_token(float(d))}A"
                    f"_{len(atoms)}atoms"
                    f"_M{safe_token(metal_abs)}pct"
                    f"_MoS2{safe_token(mos2_abs)}pct"
                )

                files = {}
                if write_structures:
                    files = write_all_formats(atoms, outdir, stem)

                rec = dict(base)
                rec.update({
                    "interlayer_distance_A": float(d),
                    "actual_atoms": len(atoms),
                    "stem": stem,
                    "vasp": files.get("vasp"),
                    "cif": files.get("cif"),
                    "xyz": files.get("xyz"),
                })
                metadata_records.append(rec)

                print(
                    f"Wrote {stem} | "
                    f"metal={metal_abs:.3f}% | MoS2={mos2_abs:.3f}%"
                )

    summary_df = pd.DataFrame(summary_records)
    metadata_df = pd.DataFrame(metadata_records)
    skipped_df = pd.DataFrame(skipped_records)

    summary_csv = outdir / "unique_targets_strain_targeted_summary.csv"
    metadata_csv = outdir / "all_generated_structures_metadata.csv"
    skipped_csv = outdir / "skipped_targets.csv"

    summary_json = outdir / "unique_targets_strain_targeted_summary.json"
    metadata_json = outdir / "all_generated_structures_metadata.json"
    skipped_json = outdir / "skipped_targets.json"

    summary_df.to_csv(summary_csv, index=False)
    metadata_df.to_csv(metadata_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)

    summary_df.to_json(summary_json, orient="records", indent=2)
    metadata_df.to_json(metadata_json, orient="records", indent=2)
    skipped_df.to_json(skipped_json, orient="records", indent=2)

    print("\nSaved:")
    print(summary_csv)
    print(metadata_csv)
    print(skipped_csv)

    return summary_df, metadata_df, skipped_df
