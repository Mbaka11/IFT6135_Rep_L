"""Provided synthetic data for Q2.

Tile labels: 0 = dark (0.2), 1 = light (0.8). Each 6x6 board has exactly
18 of each. A positive has EXACTLY ONE monochromatic 2x2 tile block;
a negative has NONE. Pattern anchors are zero-indexed top-left tile indices.

Train, validation and seen test positive anchors: (1,1), (1,3).
Unseen test positive anchors: (3,1), (3,3).
No wrap-around. Positive anchors and their light/dark types are balanced.
All four splits are class-balanced and have disjoint complete board layouts.
The two test splits use separate negatives, sampled by the same rule.
"""
from __future__ import annotations
import numpy as np

SEEN_ANCHORS = ((1, 1), (1, 3))
UNSEEN_ANCHORS = ((3, 1), (3, 3))


def block_mask(tiles: np.ndarray) -> np.ndarray:
    """Return bool (...,5,5) mask marking monochromatic 2x2 tile blocks."""
    a = tiles[..., :-1, :-1]
    return ((a == tiles[..., 1:, :-1]) & (a == tiles[..., :-1, 1:])
            & (a == tiles[..., 1:, 1:]))


def render_tiles(tiles: np.ndarray) -> np.ndarray:
    """Convert uint8 boards (N,6,6) to float32 images (N,1,24,24)."""
    intensities = np.where(tiles == 0, np.float32(0.2), np.float32(0.8))
    return np.repeat(np.repeat(intensities, 4, axis=1), 4, axis=2)[:, None]


def make_dataset(seed: int = 2026, n_train: int = 2048,
                 n_validation: int = 512, n_test_seen: int = 512,
                 n_test_unseen: int = 512) -> dict[str, dict[str, np.ndarray]]:
    """Generate all four splits deterministically by rejection sampling.

    Return a dictionary with keys train, validation, test_seen, test_unseen.
    Every split is a dictionary containing:
      x: float32 (N,1,24,24) image array
      y: int64 (N,) labels, 0/1
      tiles: uint8 (N,6,6) underlying board, 0=dark, 1=light
      layout_id: uint64 (N,) injective 36-bit row-major board encoding
      anchor: int64 (N,2) positive block location, or (-1,-1) for negatives

    Each split size must be a positive multiple of 8: half are negative;
    the remaining half are balanced across two anchors and two block types.
    Metadata is for audits/plots only; the model must receive only x.
    """
    sizes = dict(train=n_train, validation=n_validation,
                 test_seen=n_test_seen, test_unseen=n_test_unseen)
    if any(n <= 0 or n % 8 for n in sizes.values()):
        raise ValueError('Every split size must be a positive multiple of 8')
    rng = np.random.default_rng(seed)
    negative = ('negative',)
    wanted = {negative: sum(sizes.values()) // 2}
    for name, n in sizes.items():
        anchors = UNSEEN_ANCHORS if name == 'test_unseen' else SEEN_ANCHORS
        for r, c in anchors:
            for value in (0, 1):
                key = (r, c, value)
                wanted[key] = wanted.get(key, 0) + n // 8
    buckets = {key: [] for key in wanted}
    used = set()
    total_wanted = sum(wanted.values())
    total_accepted, batches = 0, 0
    weights = np.left_shift(np.uint64(1), np.arange(36, dtype=np.uint64))
    while total_accepted < total_wanted:
        batches += 1
        if batches > 10000:
            raise RuntimeError('Sampling budget exceeded; reduce split sizes')
        scores = rng.random((4096, 36))
        chosen = np.argpartition(scores, 17, axis=1)[:, :18]
        boards = np.zeros((4096, 36), dtype=np.uint8)
        np.put_along_axis(boards, chosen, 1, axis=1)
        boards = boards.reshape(-1, 6, 6)
        mask = block_mask(boards)
        count = mask.sum(axis=(1, 2))
        for i in np.flatnonzero(count <= 1):
            if count[i] == 0:
                key, anchor = negative, (-1, -1)
            else:
                pos = int(mask[i].argmax())
                r, c = divmod(pos, 5)
                key, anchor = (r, c, int(boards[i, r, c])), (r, c)
            if key not in wanted or len(buckets[key]) >= wanted[key]:
                continue
            uid = int((boards[i].reshape(36).astype(np.uint64) * weights).sum())
            if uid in used:
                continue
            used.add(uid)
            buckets[key].append((boards[i].copy(), uid, anchor))
            total_accepted += 1
    for key in buckets:
        rng.shuffle(buckets[key])
    cursor = {key: 0 for key in wanted}
    data = {}
    for name, n in sizes.items():
        anchors = UNSEEN_ANCHORS if name == 'test_unseen' else SEEN_ANCHORS
        records = []
        quotas = {negative: n // 2}
        quotas.update({(r, c, value): n // 8
                       for r, c in anchors for value in (0, 1)})
        for key, amount in quotas.items():
            start = cursor[key]
            records.extend(buckets[key][start:start + amount])
            cursor[key] += amount
        order = rng.permutation(n)
        records = [records[i] for i in order]
        tiles = np.stack([record[0] for record in records])
        uid = np.array([record[1] for record in records], dtype=np.uint64)
        anchor = np.array([record[2] for record in records], dtype=np.int64)
        y = (anchor[:, 0] >= 0).astype(np.int64)
        data[name] = dict(x=render_tiles(tiles), y=y, tiles=tiles,
                          layout_id=uid, anchor=anchor)
    return data


def audit_dataset(data: dict) -> dict:
    """Validate all published constraints; return a JSON-friendly audit summary.

    Raise AssertionError on a violated constraint. No model information is used.
    """
    summary, all_ids = {}, set()
    for name, split in data.items():
        tiles, y, ids = split['tiles'], split['y'], split['layout_id']
        n = len(y)
        assert tiles.shape == (n, 6, 6)
        assert set(np.unique(tiles)) <= {0, 1}
        assert np.all(tiles.sum(axis=(1, 2)) == 18)
        assert np.array_equal(split['x'], render_tiles(tiles))
        counts = block_mask(tiles).sum(axis=(1, 2))
        assert np.all(counts == y), 'Each positive needs exactly one block'
        assert int(y.sum()) * 2 == n
        current = set(map(int, ids))
        assert len(current) == n and not current.intersection(all_ids)
        all_ids.update(current)
        anchors = UNSEEN_ANCHORS if name == 'test_unseen' else SEEN_ANCHORS
        groups = {}
        for r, c in anchors:
            for value in (0, 1):
                selected = (y == 1) & (split['anchor'][:, 0] == r)
                selected &= (split['anchor'][:, 1] == c) & (tiles[:, r, c] == value)
                assert int(selected.sum()) == n // 8
                groups[f'{r},{c},type={value}'] = int(selected.sum())
        assert np.all(split['anchor'][y == 0] == -1)
        summary[name] = dict(size=n, negative=int((y == 0).sum()),
                              positive=int(y.sum()), positive_groups=groups)
    summary['unique_layouts_total'] = len(all_ids)
    return summary
