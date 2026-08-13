# Task 1 report

Status: DONE

Changed files:

- `tests/test_sparse_task2_manifest.py`
- `.superpowers/sdd/2026-08-13-sparse-task2-redesign/task-1-report.md`

Tests run:

- `$env:PYTHONPATH='E:\RadioFlow-worktrees\multiconfig-srm-01x'; & 'D:\Anaconda3\envs\radioflow-win\python.exe' -m pytest tests/test_sparse_task2_manifest.py -q`
  - First run: timed out after 120s
  - Second run: `9 passed in 144.03s (0:02:24)`
- `$env:PYTHONPATH='E:\RadioFlow-worktrees\multiconfig-srm-01x'; & 'D:\Anaconda3\envs\radioflow-win\python.exe' -m pytest tests/test_same_frequency_manifest.py -q`
  - Result: `2 passed in 9.25s`

Notes:

- The test fixture now copies the exact locked split bytes from `E:\datasets\MultiConfigRadiomap\manifests\scene_split_seed42.json`; if that file is absent, the fixture skips with a clear reason.
- The immutable-output test now asserts `ExistingSchemaMismatchError` instead of a broad `Exception`.
- Validation keeps `sample_count=819` as locked protocol metadata and separately enforces `800` manifest records from the `560/80/160` scene split.

Concerns:

- None blocking. The only slowdown was the sparse manifest test run, which needed a longer timeout to finish.
