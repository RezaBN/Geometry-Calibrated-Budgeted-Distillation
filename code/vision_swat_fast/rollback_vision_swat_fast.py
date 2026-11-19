#!/usr/bin/env python3
"""Restore the exact source files saved immediately before swat-fast-v1."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('project_root', type=Path)
a = p.parse_args()
root = a.project_root.resolve()
files = [root / 'gcbd_v2/common/cache.py', root / 'gcbd_v2/integrations/vision.py']
backups = [Path(str(f) + '.before_swat_fast_v1') for f in files]
if not all(f.is_file() for f in backups):
    raise SystemExit('Backups missing. No files changed.')
for file, backup in zip(files, backups):
    shutil.copy2(backup, file)
    print('RESTORED:', file)
fast = root / 'gcbd_v2/integrations/vision_fast.py'
if fast.exists():
    fast.unlink()
    print('REMOVED:', fast)
print('VISION SWAT OPTIMIZATION ROLLBACK: PASS')
