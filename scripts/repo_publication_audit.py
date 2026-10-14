"""Check for obvious sensitive/generated files in Git's staged file set.

No scanner can guarantee a repository contains no secrets. Always manually
review `git diff --cached --name-only` and your source before publishing.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
result = subprocess.run(
    ['git', '-C', str(ROOT), 'diff', '--cached', '--name-only', '-z'],
    check=True, stdout=subprocess.PIPE,
)
staged = [name for name in result.stdout.decode().split('\0') if name]
forbidden = {'.env', 'orchestration/db.env', 'mlflow_tracking.db'}
forbidden_dirs = ('.venv/', 'data/', 'reports/', 'orchestration/runs/',
                  'mlruns/', 'mlartifacts/')
problems = [
    name for name in staged
    if name in forbidden or name.startswith(forbidden_dirs)
]
large = [
    name for name in staged
    if (ROOT / name).is_file() and (ROOT / name).stat().st_size > 10_000_000
]
print('Staged files:', len(staged))
print('Protected path violations:', problems)
print('Files over 10 MB (review):', large)
if problems or large:
    sys.exit('REVIEW REQUIRED — do not push yet')
print('Staged-path check: PASS (manual secret review still required)')
