#!/usr/bin/env python3
"""Analyze strain in an already built 2D heterostructure.

Use this when you have a final heterostructure POSCAR/CIF and either:
1. saved unstrained film/substrate supercell structures, or
2. primitive film/substrate structures plus the 2x2 supercell transforms used
   to build the heterostructure.

Examples
--------
python analyze_built_heterostructure_strain.py \
  --hetero POSCAR_hetero \
  --film-ref POSCAR_film_unstrained_supercell \
  --substrate-ref POSCAR_sub_unstrained_supercell \
  --csv hetero_strain.csv --json hetero_strain.json

python analyze_built_heterostructure_strain.py \
  --hetero POSCAR_hetero \
  --film-primitive POSCAR_film --film-transform "3 0 0 3" \
  --substrate-primitive POSCAR_sub --substrate-transform "2 0 0 2"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strain_utils_2d import (
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
        "reference", "target", "eps_a_percent", "eps_b_percent", "delta_gamma_deg",
        "area_strain_percent", "principal_strain_1_percent", "principal_strain_2_percent",
        "rms_principal_strain_percent", "max_abs_principal_strain_percent",
    ]
    print(df[cols].to_string(index=False))
    print(f"\nWrote: {Path(args.csv).resolve()}")
    print(f"Wrote: {Path(args.json).resolve()}")


if __name__ == "__main__":
    main()
