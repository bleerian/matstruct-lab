# 2D heterostructure lattice mismatch / strain tools

These scripts compute in-plane strain for 2D heterostructures made with `matstruct_lab.lattice_match.find_2d_matches` and `matstruct_lab.heterostructures.build_heterostructure`.

## Install requirements

```bash
pip install numpy pandas pymatgen
# plus your local/environment install of matstruct_lab
```

## 1. Analyze match candidates before building

Edit `MATCH_KWARGS` in `analyze_matstruct_matches.py` to match your `find_2d_matches` settings, then run:

```bash
python analyze_matstruct_matches.py --film POSCAR_film --substrate POSCAR_substrate --debug-attrs
```

It writes:

- `lattice_match_strain.csv`
- `lattice_match_strain.json`

If the script cannot infer field names from the match object, `--debug-attrs` prints the available attributes. Add the relevant names to `FILM_TRANSFORM_NAMES`, `SUB_TRANSFORM_NAMES`, `FILM_VECTOR_NAMES`, or `SUB_VECTOR_NAMES`.

## 2. Analyze the final built heterostructure

Best practice is to save each layer's unstrained matched supercell before calling `build_heterostructure`, then compare those cells with the final heterostructure cell:

```bash
python analyze_built_heterostructure_strain.py \
  --hetero POSCAR_hetero \
  --film-ref POSCAR_film_unstrained_supercell \
  --substrate-ref POSCAR_substrate_unstrained_supercell
```

Alternatively, provide primitive structures and the 2x2 supercell transforms:

```bash
python analyze_built_heterostructure_strain.py \
  --hetero POSCAR_hetero \
  --film-primitive POSCAR_film --film-transform "3 0 0 3" \
  --substrate-primitive POSCAR_substrate --substrate-transform "2 0 0 2"
```

## Interpreting signs

A positive strain means the reference layer must be stretched to match the target cell. A negative strain means it must be compressed.

For a substrate-fixed model, use the `film_unstrained -> final_heterostructure_cell` row or the `film_supercell -> substrate_supercell` row.

## Reported quantities

- `eps_a_percent`, `eps_b_percent`: length mismatch along the matched in-plane lattice vectors.
- `delta_gamma_deg`: in-plane angle change.
- `area_strain_percent`: area mismatch/strain.
- `principal_strain_*_percent`: rotation-free principal in-plane strain from polar decomposition.
- `rms_principal_strain_percent`: useful scalar ranking metric.
- `max_abs_principal_strain_percent`: worst principal strain component.
