"""Checkpoint cleanup utilities: disk-space protection and old-checkpoint pruning."""

import logging
import os
import re
import shutil

logger = logging.getLogger(__name__)

_ITER_DIR_RE = re.compile(r"^iter_(\d+)$")


def _get_iter_dirs(save_dir: str) -> list[tuple[int, str]]:
    """Return sorted list of (iteration, full_path) for iter_* dirs under save_dir."""
    results = []
    if not os.path.isdir(save_dir):
        return results
    for name in os.listdir(save_dir):
        m = _ITER_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(save_dir, name)):
            results.append((int(m.group(1)), os.path.join(save_dir, name)))
    results.sort(key=lambda x: x[0])
    return results


def _dir_size(path: str) -> int:
    """Return total size in bytes of a directory tree."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def cleanup_old_checkpoints(save_dir: str, max_keep: int = 1) -> None:
    """Delete old checkpoints, keeping only the newest *max_keep*.

    Scans *save_dir* for ``iter_NNNNNNN`` directories, sorts by iteration
    number, and removes the oldest ones until at most *max_keep* remain.
    """
    if max_keep < 1:
        max_keep = 1

    iter_dirs = _get_iter_dirs(save_dir)
    if len(iter_dirs) <= max_keep:
        return

    to_delete = iter_dirs[: len(iter_dirs) - max_keep]
    for iteration, path in to_delete:
        logger.info("Checkpoint cleanup: removing old checkpoint iter_%07d (%s)", iteration, path)
        shutil.rmtree(path, ignore_errors=True)

    logger.info(
        "Checkpoint cleanup: deleted %d checkpoint(s), kept %d",
        len(to_delete),
        max_keep,
    )


def check_disk_space_and_cleanup(save_dir: str, min_free_bytes: int | None = None) -> None:
    """Ensure enough disk space before writing a new checkpoint.

    Strategy:
    - Estimate single-checkpoint size from the newest existing checkpoint.
    - If free space < 2 * ckpt_size (or *min_free_bytes* if given), delete
      the oldest checkpoints one-by-one until the condition is met.
    - Always keeps at least 1 checkpoint.
    """
    if not os.path.isdir(save_dir):
        return

    iter_dirs = _get_iter_dirs(save_dir)
    if not iter_dirs:
        return

    # Estimate checkpoint size from the newest one.
    newest_path = iter_dirs[-1][1]
    ckpt_size = _dir_size(newest_path)
    if ckpt_size == 0:
        return

    threshold = min_free_bytes if min_free_bytes is not None else 2 * ckpt_size

    deleted = 0
    for iteration, path in iter_dirs[:-1]:  # never delete the newest
        try:
            free = shutil.disk_usage(save_dir).free
        except OSError:
            return
        if free >= threshold:
            break
        logger.warning(
            "Disk space low (%.1f GB free, need %.1f GB). Removing checkpoint iter_%07d",
            free / (1024**3),
            threshold / (1024**3),
            iteration,
        )
        shutil.rmtree(path, ignore_errors=True)
        deleted += 1

    if deleted:
        logger.info("Disk-space cleanup: removed %d old checkpoint(s)", deleted)
