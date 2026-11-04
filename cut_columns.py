"""
cut_columns.py - cut flybrain down to a patch of ~150 optic lobe columns.

What it does:
  1. Asks neuPrint which column each optic lobe neuron belongs to.
  2. Picks the ~150 columns closest to the middle of the right eye.
  3. Keeps the neurons in those columns, plus the "partner" neurons that
     send or receive a big share of their connections inside the patch.
  4. Cuts the connection matrix down to just those neurons and saves it.

Run it on your PC, not the Pi. The full matrix needs a few GB of RAM.
"""

import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ---------------- Settings you can change ----------------
N_COLUMNS = 150      # how many columns to keep
SIDE = "R"           # which optic lobe (the right one is the best proofread)
MIN_SHARE = 0.2      # keep a partner if >= 20% of its connections touch the patch

DATA = Path.home() / "fly-data"
COLUMNS_CACHE = Path("column_table.csv")   # saved so we only ask neuPrint once
OUT_BRAIN = Path("patch_brain.npz")
OUT_WEIGHTS = Path("patch_weights.npz")


# ---------------- Step 1: which neuron is in which column ----------------
def fetch_column_table():
    """Return a table with one row per column-assigned neuron."""
    import pandas as pd

    if COLUMNS_CACHE.exists():
        print(f"Using saved column table: {COLUMNS_CACHE}")
        return pd.read_csv(COLUMNS_CACHE)

    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit("Set your neuPrint token first:  export NEUPRINT_TOKEN='...'")

    from neuprint import Client
    client = Client("neuprint.janelia.org", dataset="male-cns:v1.0", token=token)

    # Ask the database for every neuron that has a column assigned to it.
    query = """
        MATCH (n:Neuron)
        WHERE n.assignedOlHex1 IS NOT NULL
        RETURN n.bodyId AS body_id, n.type AS type, n.somaSide AS side,
               n.assignedOlHex1 AS hex1, n.assignedOlHex2 AS hex2
    """
    table = client.fetch_custom(query)
    if len(table) == 0:
        sys.exit("neuPrint returned no column assignments. The property name may "
                 "differ in this dataset. Paste this message to Claude.")
    table.to_csv(COLUMNS_CACHE, index=False)
    print(f"Saved {len(table):,} column assignments to {COLUMNS_CACHE}")
    return table


# ---------------- Step 2: load flybrain's connection matrix ----------------
def load_weights(path, n):
    """Load weights.npz as a sparse matrix, whichever common layout it uses."""
    f = np.load(path)
    keys = set(f.files)
    if {"data", "indices", "indptr"} <= keys:
        layout = str(f["format"]) if "format" in keys else "csr"
        shape = tuple(f["shape"]) if "shape" in keys else (n, n)
        make = sp.csc_matrix if "csc" in layout else sp.csr_matrix
        return make((f["data"], f["indices"], f["indptr"]), shape=shape).tocsr()
    for a, b, w in [("pre", "post", "weight"), ("src", "dst", "weight"),
                    ("row", "col", "data")]:
        if {a, b, w} <= keys:
            return sp.coo_matrix((f[w], (f[a], f[b])), shape=(n, n)).tocsr()
    sys.exit(f"I don't recognize the layout of {path}. Arrays inside: {f.files}. "
             "Paste this message to Claude.")


def main():
    brain = np.load(DATA / "brain.npz")
    ids = brain["ids"]                       # ids[row] = MaleCNS body ID
    n = len(ids)
    row_of = {int(b): i for i, b in enumerate(ids)}
    print(f"flybrain has {n:,} neurons")

    # ---- Pick the columns ----
    table = fetch_column_table()
    if "side" in table and table["side"].notna().any():
        table = table[table["side"] == SIDE]
    table = table[table["body_id"].isin(row_of.keys())]   # only neurons flybrain has
    print(f"{len(table):,} column-assigned neurons on side {SIDE}")

    columns = table[["hex1", "hex2"]].drop_duplicates().to_numpy(dtype=float)
    middle = np.median(columns, axis=0)
    distance = np.hypot(*(columns - middle).T)             # distance from the middle
    chosen = columns[np.argsort(distance)[:N_COLUMNS]]
    chosen_set = {tuple(c) for c in chosen}
    print(f"Chose {len(chosen)} columns around hex ({middle[0]:.0f}, {middle[1]:.0f})")

    in_patch = [tuple(c) in chosen_set for c in table[["hex1", "hex2"]].to_numpy(float)]
    core = np.array(sorted(row_of[int(b)] for b in table.loc[in_patch, "body_id"]))
    print(f"Core: {len(core):,} neurons inside the chosen columns")

    # ---- Find partners ----
    print("Loading connections (this takes a moment)...")
    W = load_weights(DATA / "weights.npz", n)
    strength = abs(W) + abs(W).T        # counts connections in both directions
    core_mask = np.zeros(n)
    core_mask[core] = 1.0
    to_patch = strength @ core_mask      # how much of each neuron touches the core
    total = np.asarray(strength.sum(axis=1)).ravel()
    share = np.divide(to_patch, total, out=np.zeros(n), where=total > 0)
    partners = np.where((share >= MIN_SHARE) & (core_mask == 0))[0]
    print(f"Partners: {len(partners):,} neurons with >= {MIN_SHARE:.0%} "
          "of their connections in the patch")

    # ---- Cut and save ----
    keep = np.union1d(core, partners)
    W_patch = W[keep][:, keep]
    sp.save_npz(OUT_WEIGHTS, W_patch)

    saved = {"original_rows": keep, "is_core": np.isin(keep, core)}
    for name in brain.files:              # slice every per-neuron array in brain.npz
        arr = brain[name]
        saved[name] = arr[keep] if arr.ndim > 0 and len(arr) == n else arr
    np.savez(OUT_BRAIN, **saved)

    kept_fraction = W_patch.nnz / W.nnz
    print(f"\nKept {len(keep):,} of {n:,} neurons "
          f"and {W_patch.nnz:,} of {W.nnz:,} connections ({kept_fraction:.2%})")
    print(f"Rough speedup per step: about {1 / kept_fraction:.0f}x")
    print(f"Saved {OUT_BRAIN} and {OUT_WEIGHTS}")


if __name__ == "__main__":
    main()
