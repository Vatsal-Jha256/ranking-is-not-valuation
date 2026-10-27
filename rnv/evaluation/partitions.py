"""Manufacturer partitions: how a held-out supplier is defined.

Two constructors live here, and only one of them is used by anything the manuscript
claims.

:func:`stratified_random_group_blocks` is the accepted one. Manufacturer sizes in the
registry are heavily skewed -- a handful of producers publish tens of thousands of
declarations -- so a partition has to keep whole manufacturers together while giving
each block a comparable size profile, and it has to genuinely move when the seed
moves. It does both: manufacturers are ordered by size, and each consecutive chunk of
``n_blocks`` of them is dealt out to the blocks in a seeded random order. The returned
digest fingerprints the whole assignment, which is what lets a result ledger prove
which partition produced it.

:func:`balanced_group_blocks` is greedy largest-first bin packing. It balances rows
well but uses its seed only to break ties between manufacturers of identical size, so
different seeds return near-identical blocks. That is why it is not used for anything
reported: it cannot support a claim about sensitivity across partitions. It is kept
because two supplementary experiments in ``experiments/supplementary`` and the pilot
were built on it, and rewriting them would misrepresent what was run.

Supplies: the evaluation-protocol paragraph of the manuscript and Appendix B; the
partition-overlap diagnostic reported alongside the headline table.
"""
from __future__ import annotations

import hashlib

import numpy as np


def stratified_random_group_blocks(groups, n_blocks: int, seed: int):
    """Seeded, size-stratified manufacturer partition with balanced size profiles.

    Args:
        groups: manufacturer label per row.
        n_blocks: number of blocks to deal manufacturers into.
        seed: partition seed; distinct seeds give genuinely distinct partitions.

    Returns:
        ``(blocks, loads, digest)`` -- a block id per row, the row count per block,
        and a sha256 over the manufacturer-to-block assignment.
    """
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    rng = np.random.default_rng(seed)
    assignment = {}
    for start in range(0, len(order), n_blocks):
        chunk = order[start:start + n_blocks]
        labels = rng.choice(n_blocks, size=len(chunk), replace=False)
        for group_index, label in zip(chunk, labels):
            assignment[unique[group_index]] = int(label)
    blocks = np.asarray([assignment[group] for group in groups], dtype=int)
    loads = np.bincount(blocks, minlength=n_blocks)
    digest_text = "\\n".join(
        f"{group}\\t{assignment[group]}" for group in sorted(assignment, key=str)
    )
    digest = hashlib.sha256(digest_text.encode("utf-8")).hexdigest()
    return blocks, loads, digest


def balanced_group_blocks(groups, n_blocks=3, weights=None, seed=0):
    """Greedy largest-first packing of whole manufacturers into row-balanced blocks.

    Superseded for anything the manuscript reports; see the module docstring.

    Args:
        groups: manufacturer label per row.
        n_blocks: number of blocks.
        weights: optional per-manufacturer weights in place of row counts.
        seed: only breaks ties between equal-sized manufacturers.

    Returns:
        ``(blocks, loads)`` -- a block id per row and the accumulated load per block.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    if weights is not None:
        counts = np.asarray(weights)

    rng = np.random.default_rng(seed)
    jitter = rng.random(len(uniq))
    order = np.lexsort((jitter, -counts))     # largest first, ties broken randomly

    loads = np.zeros(n_blocks, dtype=float)
    assign = {}
    for i in order:
        b = int(np.argmin(loads))
        assign[uniq[i]] = b
        loads[b] += counts[i]

    out = np.array([assign[g] for g in groups], dtype=int)
    return out, loads


def kfold_group_blocks(groups, n_folds=5, seed=0):
    """Balanced group k-fold; returns a fold id per row."""
    return balanced_group_blocks(groups, n_blocks=n_folds, seed=seed)[0]
