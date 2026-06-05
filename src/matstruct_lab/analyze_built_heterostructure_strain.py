#!/usr/bin/env python3
"""Analyze strain in an already built 2D heterostructure.

This is a small CLI wrapper around matstruct_lab.strain_utils_2d kept for older
workflows that pass pymatgen-readable structure files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from matstruct_lab.strain_utils_2d import (
    cell2d_from_structure,
    load_structure,
    parse_matrix,
    strain_summary,
    to_plain_dict,
    transform_2d_cell,
    write_json,
)


def ref_cell_from_args(ref_path: str | None, prim_path: str | None, transform_text: str | None):
    if ref_path:
        return cell2d_from_structure(load_structure(ref_path))
    if prim_path and transform_text:
        prim_cell = cell2d_from_structure(load_structure(prim_path))
        transform = parse_matrix(transform_text)
        return transform_2d_cell(prim_cell, transform)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hetero", required=True, help="Final built heterostructure file, e.g. POSCAR or CIF")
    p.add_argument("--film-ref", help="Unstrained film supercell used before final common-cell strain")
    p.add_argument("--substrate-ref", help="Unstrained substrate supercell used before final common-cell strain")
    p.add_argument("--film-primitive", help="Primitive film structure; requires --film-transform")
    p.add_argument("--film-transform", help="2x2 film transform, e.g. '3 0 0 3' or '[[3,0],[0,3]]'")
    p.add_argument("--substrate-primitive", help="Primitive substrate structure; requires --substrate-transform")
    p.add_argument("--substrate-transform", help="2x2 substrate transform")
    p.add_argument("--csv", default="heterostructure_strain.csv")
    p.add_argument("--json", default="heterostructure_strain.json")
    args = p.parse_args()

    target_cell = cell2d_from_structure(load_structure(args.hetero))

    records = []
    film_ref = ref_cell_from_args(args.film_ref, args.film_primitive, args.film_transform)
    sub_ref = ref_cell_from_args(args.substrate_ref, args.substrate_primitive, args.substrate_transform)

    if film_ref is not None:
        records.append(to_plain_dict(strain_summary(film_ref, target_cell, "film_unstrained", "final_heterostructure_cell")))
    if sub_ref is not None:
        records.append(to_plain_dict(strain_summary(sub_ref, target_cell, "substrate_unstrained", "final_heterostructure_cell")))

    if not records:
        raise SystemExit("No layer reference supplied. Provide --film-ref/--substrate-ref or primitive + transform.")

    df = pd.DataFrame(records)
    df.to_csv(args.csv, index=False)
    write_json(args.json, records)

    cols = [
        "reference",
        "target",
        "eps_a_percent",
        "eps_b_percent",
        "delta_gamma_deg",
        "area_strain_percent",
        "principal_strain_1_percent",
        "principal_strain_2_percent",
        "rms_principal_strain_percent",
        "max_abs_principal_strain_percent",
    ]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\nWrote: {Path(args.csv).resolve()}")
    print(f"Wrote: {Path(args.json).resolve()}")


if __name__ == "__main__":
    main()
