#!/usr/bin/env python3
"""Apply versioned Vision-only SWAT performance patch, with safe rollback.

Usage: python apply_vision_swat_fast.py /scratch/.../gcbd_phase2_integrated
Must be run from the bundled directory so vision_fast.py and verify_fast_swat.py
are present. Does NOT touch checkpoints, caches, or experiment configurations.
"""
from __future__ import annotations

import argparse
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SUFFIX = '.before_swat_fast_v1'


def replace_atomically(path: Path, content: str):
    fd, temp_path = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_root', type=Path)
    parser.add_argument('--skip-cpu-check', action='store_true',
                        help='Skip CPU reference test only (not recommended)')
    args = parser.parse_args()
    root = args.project_root.resolve()
    here = Path(__file__).resolve().parent
    cache = root / 'gcbd_v2/common/cache.py'
    vision = root / 'gcbd_v2/integrations/vision.py'
    fast = root / 'gcbd_v2/integrations/vision_fast.py'
    src_fast = here / 'vision_fast.py'
    verification = here / 'verify_fast_swat.py'

    if not (cache.is_file() and vision.is_file() and src_fast.is_file() and verification.is_file()):
        raise SystemExit('Missing source or bundled files; no changes made.')
    if fast.exists():
        raise SystemExit(f'Already present: {fast}. No changes made.')
    targets = (cache, vision)
    backups = [Path(str(p) + SUFFIX) for p in targets]
    if any(p.exists() for p in backups):
        raise SystemExit('A prior SWAT-fast backup exists. Refusing re-application.')

    old_get = '''    def get_many(self, sample_ids: Iterable[int]) -> list[dict[str, Any]]:
        return [self.get(int(x)) for x in sample_ids]
'''
    new_get = '''    def get_many(self, sample_ids: Iterable[int]) -> list[dict[str, Any]]:
        """Fetch a minibatch in shard groups, preserving requested sample order.

        Loading each shard once per call avoids pathological LRU churn when a
        shuffled minibatch spans many disk shards. Checksum verification and
        shared-support reconstruction are unchanged from :meth:`get`.
        """
        ids = [int(x) for x in sample_ids]
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for position, sample_id in enumerate(ids):
            if sample_id not in self._location_by_sample:
                raise KeyError(f"Sample ID {sample_id} is absent from geometry cache")
            shard_id, row = self._location_by_sample[sample_id]
            groups.setdefault(shard_id, []).append((position, sample_id, row))
        results: list[dict[str, Any] | None] = [None] * len(ids)
        for shard_id, items in groups.items():
            payload = self._load(shard_id)
            shared_supports = payload.get("shared_supports", {})
            for position, sample_id, row in items:
                if row >= len(payload["records"]) or int(payload["sample_ids"][row]) != sample_id:
                    raise RuntimeError("Sample index points to an inconsistent shard record")
                record = payload["records"][row]
                if shared_supports and isinstance(record, dict) and isinstance(record.get("layers"), dict):
                    materialized_layers = {}
                    for layer_name, layer_record in record["layers"].items():
                        if "support_ref" in layer_record:
                            ref = layer_record["support_ref"]
                            if ref not in shared_supports:
                                raise RuntimeError(f"Missing shared support {ref!r} in cache shard {shard_id}")
                            materialized_layers[layer_name] = {**layer_record, "support": shared_supports[ref]}
                        else:
                            materialized_layers[layer_name] = layer_record
                    record = {**record, "layers": materialized_layers}
                results[position] = record
        return results
'''
    original_cache = cache.read_text(encoding='utf-8')
    original_vision = vision.read_text(encoding='utf-8')
    if original_cache.count(old_get) != 1:
        raise SystemExit('Cache reader source anchor differs; no changes made.')
    if original_vision.count('def cached_vision_swat_loss(') != 1 or \
       original_vision.count('def queried_vision_final_swat_loss(') != 1:
        raise SystemExit('Vision reference functions not found; no changes made.')
    if 'cached_vision_swat_loss_fast as cached_vision_swat_loss' in original_vision:
        raise SystemExit('Vision optimized aliases already installed; no changes made.')

    updated_cache = original_cache.replace(old_get, new_get)
    updated_vision = original_vision.rstrip() + '''\n\n\n# Vision-only vectorized implementation. Original reference functions remain
# above for loss/gradient equivalence testing and reproducibility.
from .vision_fast import (
    cached_vision_swat_loss_fast as cached_vision_swat_loss,
    queried_vision_final_swat_loss_fast as queried_vision_final_swat_loss,
)
'''
    # Parse all proposed source files before changing anything.
    import ast
    for contents, filename in ((updated_cache, str(cache)),
                               (updated_vision, str(vision)),
                               (src_fast.read_text(), str(fast))):
        ast.parse(contents, filename=filename)
    for src, dest in zip(targets, backups):
        shutil.copy2(src, dest)

    try:
        replace_atomically(cache, updated_cache)
        replace_atomically(vision, updated_vision)
        shutil.copy2(src_fast, fast)
        for item in (cache, vision, fast):
            py_compile.compile(str(item), doraise=True)
        if not args.skip_cpu_check:
            env = dict(os.environ)
            env['PYTHONPATH'] = f'{root}:{root / "vision"}' + (
                ':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            print('Running CPU loss/gradient reference check...', flush=True)
            subprocess.run([sys.executable, str(verification), '--device', 'cpu'],
                           check=True, env=env)
    except BaseException:
        print('PATCH VALIDATION FAILED: Restoring original source files.', file=sys.stderr)
        for dest, backup in zip(targets, backups):
            if backup.is_file():
                shutil.copy2(backup, dest)
        if fast.exists():
            fast.unlink()
        raise
    print('VISION SWAT OPTIMIZATION PATCH: PASS')
    print('Unmodified checkpoint and cache formats: PASS')
    for target, backup in zip(targets, backups):
        print(f'Patched: {target}\nBackup:  {backup}')
    print(f'New module: {fast}')


if __name__ == '__main__':
    main()
