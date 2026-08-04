# Multi-Config SRM Array Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the pinned Hxxxz0/RadioFlow checkout with a reproducible adapter for the arXiv:2603.06401 Multi-config Radiomap dataset, then train and evaluate six independent 6.7 GHz SRM models for 8x8, 16x16, and 32x32 arrays in Lite and Large sizes.

**Architecture:** Keep RadioFlow's `DiffUNet`, `BasicUNetEncoder`, `BasicUNetDe`, conditional flow-matching objective, EMA rule, CFG implementation, and two-step Euler generation path. Add separate dataset preparation, manifest, masked loss/metric, checkpoint, training, and evaluation modules around the unchanged learned architecture. The only model-side change is an execution-only activation-checkpoint switch for the locked Large network; tests must prove that it adds no parameters or state keys and preserves outputs and gradients.

**Tech Stack:** Python 3.10, PyTorch 2.5.1+cu121, MONAI, torchcfm, NumPy, scikit-image, Matplotlib, Hugging Face Hub, pytest, PowerShell, NVIDIA CUDA on the RTX A2000 Laptop GPU.

## Global Constraints

- Work in `E:\RadioFlow`, whose `origin` must remain `https://github.com/Hxxxz0/RadioFlow.git` and whose history must descend from `8944e3160f6a7a85b5451ae58e337186a4d98771`.
- Preserve `train.py`, `test.py`, and `data_loaders/loaders.py` as the original RadioMapSeer entry points. The Multi-config benchmark uses new entry points and must never import a learned model from the arXiv:2603.06401 reference repository.
- Reuse `model.model.DiffUNet`, `model.unet.basic_unet.BasicUNetEncoder`, `model.unet.basic_unet_denose.BasicUNetDe`, `torchcfm.conditional_flow_matching.ConditionalFlowMatcher`, `train.ModelEMA`, and `DiffUNet.forward_with_cfg`.
- Lock three condition channels, 256x256 output, 6.7 GHz, seed 42, eight common angles `[-28, -21, -14, -7, 0, 7, 14, 21]`, and scene-disjoint 560/80/160 splits. Do not expose command-line overrides for those scientific controls.
- Lock feature tuples to Lite `(32, 32, 64, 128, 256, 32)` and Large `(128, 128, 256, 512, 1024, 128)`, with three-channel parameter counts 3,994,859 and 54,126,059.
- Use the dataset source revision `49ca1dcebe2caa2b2112e6c862132243a992b00a`, archive `Dataset_20260306164917.zip`, and reference-code revision `f64e22a578933aa0ba57850ab2c7cf0695063c90`.
- Do not infer array rows and columns from `sqrt(tx_elements)`. Audit the real configuration files first, cross-check the pinned official generator when required, and commit the resulting schema lock before building manifests.
- Keep downloaded data under `E:\datasets\MultiConfigRadiomap`, training state under `E:\RadioFlow\runs\srm_6.7ghz_common8`, and evaluation output under `E:\RadioFlow\results\srm_6.7ghz_common8`. Never commit archives, extracted arrays, checkpoints, predictions, or result images.
- Preserve the user's untracked `.llm-chat-history/` and `.vscode/` directories. Stage only the files named by the current task.
- Use `D:\Anaconda3\envs\radioflow-win\python.exe` for every Python and pytest command. Run `git diff --check` and the task-specific tests before every commit.
- Follow red-green-refactor for code tasks: add the stated test, run it and confirm the expected missing-behavior failure, implement the smallest complete behavior, rerun the focused test, then rerun all earlier relevant tests.
- Fail closed on missing data, ambiguous configuration, invalid sentinels, non-finite tensors, empty masks, identity mismatch, corrupt checkpoints, and incomplete evaluation output. Evaluation must never fall back to random weights.

### Fixed artifact layout

```text
E:\datasets\MultiConfigRadiomap\
  downloads\Dataset_20260306164917.zip
  download_receipt.json
  extraction_receipt.json
  reference_code\
  raw\
  manifests\scene_split_seed42.json
  manifests\manifest_8x8.jsonl
  manifests\manifest_16x16.jsonl
  manifests\manifest_32x32.jsonl
  manifests\height_stats_seed42.json
  manifests\visualization_cases_seed42.json

E:\RadioFlow\runs\srm_6.7ghz_common8\<array>\<model_size>\
  best.pt
  last.pt
  config.json
  metrics.csv
  training_runtime.json
  cfg_selection.json

E:\RadioFlow\runs\srm_6.7ghz_common8\_hardware\
  large_hardware_gate.json

E:\RadioFlow\results\srm_6.7ghz_common8\<array>\<model_size>\
  cfg_selection.json
  metrics_test.json
  metrics_per_beam.csv
  runtime.json
  run_manifest.json
  predictions\
  comparisons\
  error_maps\
```

---

### Task 1: Establish the test harness and lock RadioFlow provenance

**Files:**

- Create: `pytest.ini`
- Create: `requirements-multiconfig.txt`
- Create: `experiments/__init__.py`
- Create: `training/__init__.py`
- Create: `evaluation/__init__.py`
- Create: `experiments/provenance.py`
- Create: `tests/test_radioflow_framework_lock.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add a failing framework-lock test**

  Write tests that instantiate both three-channel models and assert the exact feature tuples, parameter counts, concrete encoder/decoder classes, Git origin, and baseline ancestry. The core assertions are:

  ```python
  from config import MODEL_FEATURES
  from model.model import DiffUNet
  from model.unet.basic_unet import BasicUNetEncoder
  from model.unet.basic_unet_denose import BasicUNetDe

  EXPECTED_COUNTS = {"lite": 3_994_859, "large": 54_126_059}

  def test_locked_radioflow_model_family():
      assert MODEL_FEATURES["lite"] == (32, 32, 64, 128, 256, 32)
      assert MODEL_FEATURES["large"] == (128, 128, 256, 512, 1024, 128)
      for size, expected_count in EXPECTED_COUNTS.items():
          network = DiffUNet(con_channels=3, model_size=size)
          assert type(network) is DiffUNet
          assert type(network.embed_model) is BasicUNetEncoder
          assert type(network.model) is BasicUNetDe
          assert sum(p.numel() for p in network.parameters()) == expected_count
  ```

- [ ] **Step 2: Run the test and verify the intended red state**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_radioflow_framework_lock.py -q
  ```

  Expected failure: importing `experiments.provenance` fails because the provenance contract does not yet exist. The parameter assertions themselves must already agree with upstream.

- [ ] **Step 3: Implement provenance checks and test configuration**

  In `experiments/provenance.py`, define immutable source constants, `sha256_file(path)`, `git_output(repo_root, *args)`, and `assert_radioflow_checkout(repo_root)`. The checkout assertion must compare the normalized origin URL and run:

  ```python
  subprocess.run(
      ["git", "merge-base", "--is-ancestor", RADIOFLOW_UPSTREAM_BASE, "HEAD"],
      cwd=repo_root,
      check=True,
      capture_output=True,
      text=True,
  )
  ```

  Add `dataset`, `gpu`, and `slow` markers to `pytest.ini`. Add `huggingface-hub>=0.20,<2`, `pytest>=8,<10`, `torchdiffeq>=0.2.3,<0.3`, and `torchdyn>=1.0,<2` to `requirements-multiconfig.txt`; do not include or reinstall the working PyTorch/CUDA stack. Ignore `.pytest_cache/`, `runs/`, downloaded ZIP files, and generated benchmark results.

- [ ] **Step 4: Run focused and baseline tests**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pip install --only-binary=:all: --upgrade-strategy only-if-needed -r requirements-multiconfig.txt
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_radioflow_framework_lock.py -q
  D:\Anaconda3\envs\radioflow-win\python.exe -c "import torch, torchcfm, monai; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
  git diff --check
  ```

- [ ] **Step 5: Commit the harness**

  ```powershell
  git add .gitignore pytest.ini requirements-multiconfig.txt experiments/__init__.py experiments/provenance.py training/__init__.py evaluation/__init__.py tests/test_radioflow_framework_lock.py
  git commit -m "test: lock RadioFlow benchmark provenance"
  ```

---

### Task 2: Download, verify, and safely extract the pinned dataset archive

**Files:**

- Create: `experiments/multiconfig_download.py`
- Create: `prepare_multiconfig.py`
- Create: `tests/test_multiconfig_download.py`

- [ ] **Step 1: Write archive-integrity and extraction-security tests**

  Use temporary, locally generated ZIP files. Cover a valid archive with a hand-computed SHA-256, a corrupt ZIP, a mismatched expected hash, an absolute member path, a `..` traversal member, and insufficient extraction space. Use these public data structures:

  ```python
  @dataclass(frozen=True)
  class DatasetSource:
      repo_id: str
      revision: str
      filename: str

  @dataclass(frozen=True)
  class ArchiveVerification:
      filename: str
      size_bytes: int
      sha256: str
      zip_members: int
      uncompressed_bytes: int

  OFFICIAL_SOURCE = DatasetSource(
      repo_id="lxj321/Multi-config-Radiomap-Dataset",
      revision="49ca1dcebe2caa2b2112e6c862132243a992b00a",
      filename="Dataset_20260306164917.zip",
  )
  ```

- [ ] **Step 2: Confirm the tests fail because the module is absent**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_download.py -q
  ```

- [ ] **Step 3: Implement verified download and safe extraction**

  Implement these exact interfaces:

  ```python
  def verify_zip(
      archive_path: Path,
      expected_sha256: str | None = None,
  ) -> ArchiveVerification:
      verification = ArchiveVerification(
          filename=archive_path.name,
          size_bytes=archive_path.stat().st_size,
          sha256=sha256_file(archive_path),
          zip_members=0,
          uncompressed_bytes=0,
      )
      return _verify_members_and_replace_counts(
          archive_path, verification, expected_sha256
      )

  def download_archive(
      source: DatasetSource,
      downloads_dir: Path,
  ) -> tuple[Path, dict[str, object]]:
      destination = downloads_dir / source.filename
      temporary = destination.with_suffix(destination.suffix + ".part")
      url = hf_hub_url(
          repo_id=source.repo_id,
          repo_type="dataset",
          filename=source.filename,
          revision=source.revision,
      )
      _stream_url_to_file(url, temporary)
      checked = verify_zip(temporary)
      os.replace(temporary, destination)
      receipt = {
          "repo_id": source.repo_id,
          "repo_type": "dataset",
          "revision": source.revision,
          **asdict(checked),
      }
      receipt["filename"] = source.filename
      return destination, receipt

  def safe_extract_zip(archive_path: Path, destination: Path) -> Path:
      checked = verify_zip(archive_path)
      if destination.exists():
          _validate_existing_extraction_or_raise(destination, checked)
          return destination
      _assert_free_space(destination.parent, checked.uncompressed_bytes)
      _assert_all_members_within_destination(archive_path, destination)
      temporary = _extract_to_verified_temporary_directory(
          archive_path, destination.parent
      )
      temporary.rename(destination)
      return destination
  ```

  `ZipFile.testzip()` must return `None`. Resolve every member against the temporary extraction root and reject absolute paths, drive-qualified paths, `..`, symlinks, and members outside it. Require free bytes of at least `ceil(uncompressed_bytes * 1.10)`. Stream directly to the `.part` file on `E:` so the Hub cache does not create a second multi-gigabyte archive. If a previously verified destination archive and matching `download_receipt.json` exist, validate and reuse them; never replace them before the new `.part` passes all checks.

  Keep the two state transitions separate: `download` writes `download_receipt.json` immediately after the verified atomic archive publication; `extract` writes `extraction_receipt.json` only after all members and the extracted inventory validate. On Windows, rename the temporary directory only when `raw` does not exist. If `raw` already exists, reuse it only when the extraction receipt matches the archive SHA-256, member count, uncompressed byte count, and inventory hash; otherwise fail without deleting or overwriting it.

  Add `download` and `extract` subcommands to `prepare_multiconfig.py`; both use the fixed `OFFICIAL_SOURCE` rather than accepting an arbitrary revision.

- [ ] **Step 4: Run security tests and the earlier lock test**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_download.py tests/test_radioflow_framework_lock.py -q
  git diff --check
  ```

- [ ] **Step 5: Download and extract the real pinned archive**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe prepare_multiconfig.py download --dataset-root E:\datasets\MultiConfigRadiomap
  D:\Anaconda3\envs\radioflow-win\python.exe prepare_multiconfig.py extract --dataset-root E:\datasets\MultiConfigRadiomap
  ```

  Inspect `download_receipt.json`, rerun `verify_zip`, and record the archive SHA-256 and member count. If free-space validation fails, stop and report the measured requirement; do not redirect large files to `C:`.

- [ ] **Step 6: Commit only code and tests**

  ```powershell
  git add experiments/multiconfig_download.py prepare_multiconfig.py tests/test_multiconfig_download.py
  git commit -m "feat: verify pinned Multi-config dataset archive"
  ```

---

### Task 3: Audit the real configuration schema and freeze array/beam identities

**Files:**

- Create: `experiments/multiconfig_manifest.py`
- Create after real-data audit: `experiments/multiconfig_schema.json`
- Create: `tests/fixtures/multiconfig_config/` with minimal redacted format fixtures derived from the pinned archive
- Create: `tests/test_multiconfig_schema.py`
- Modify: `prepare_multiconfig.py`

- [ ] **Step 1: Inspect the real bytes and pin the reference generator**

  Before naming or parsing any configuration field, list the extracted archive and read the actual `configs/*.txt` and beam-setting bytes without changing them. Fix the effective dataset root even if the ZIP has an extra top-level directory: `workspace_root` is `E:\datasets\MultiConfigRadiomap`; `data_root` is the precise path below it that contains the released data; all manifest paths will be relative to `workspace_root` and therefore retain their `raw/...` prefix.

  Clone the official reference code outside RadioFlow and detach it at the approved commit:

  ```powershell
  git clone https://github.com/Lxj321/MulticonfigRadiomapDataset.git E:\datasets\MultiConfigRadiomap\reference_code
  git -C E:\datasets\MultiConfigRadiomap\reference_code checkout --detach f64e22a578933aa0ba57850ab2c7cf0695063c90
  git -C E:\datasets\MultiConfigRadiomap\reference_code rev-parse HEAD
  ```

  Record the SHA-256 and relative path of every reference script used to establish rows/columns or beam geometry. If `reference_code` already exists, require the exact origin and detached commit rather than fetching or checking out a moving branch.

- [ ] **Step 2: Write tests from byte-faithful real fixtures**

  Copy the smallest sufficient real configuration fragments into fixtures without renaming keys, changing separators, normalizing units, or flattening hierarchy. Then define and test the following immutable specifications:

  Define and test the following immutable specifications:

  ```python
  @dataclass(frozen=True)
  class BeamSpec:
      beam_id: int
      steering_deg: float

  @dataclass(frozen=True)
  class ArraySpec:
      name: Literal["8x8", "16x16", "32x32"]
      rows: int
      cols: int
      tx_elements: int
      frequency_hz: int
      beams: tuple[BeamSpec, ...]

  ARRAY_SPECS = {
      "8x8": ArraySpec("8x8", 8, 8, 64, 6_700_000_000,
          tuple(BeamSpec(i, -28.0 + 7.0 * i) for i in range(8))),
      "16x16": ArraySpec("16x16", 16, 16, 256, 6_700_000_000,
          tuple(BeamSpec(i, -28.0 + 3.5 * i)
                for i in (0, 2, 4, 6, 8, 10, 12, 14))),
      "32x32": ArraySpec("32x32", 32, 32, 1024, 6_700_000_000,
          tuple(BeamSpec(i, -32.0 + i)
                for i in (4, 11, 18, 25, 32, 39, 46, 53))),
  }
  ```

  Tests must prove that a synthetic `4x16, 64TR` configuration is rejected as 8x8, that all arrays yield the exact common angle tuple, and that the byte-faithful released beam-setting syntax resolves selected IDs to the required angles. Do not mention or implement a field name until Step 1 proves that exact field exists.

- [ ] **Step 3: Run the schema tests in the red state**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_schema.py -q
  ```

- [ ] **Step 4: Implement generic audit, inventory, and strict schema-lock loading**

  Implement:

  ```python
  def audit_config_files(dataset_root: Path) -> ConfigAuditReport:
      return _inventory_and_parse_all_configuration_text(dataset_root)

  def load_schema_lock(path: Path) -> DatasetSchemaLock:
      lock = DatasetSchemaLock.from_json(path.read_text(encoding="utf-8"))
      lock.validate_source_revisions()
      lock.validate_unique_configuration_ids()
      return lock

  def validate_config_against_spec(
      config: ParsedConfiguration,
      spec: ArraySpec,
  ) -> None:
      if (config.rows, config.cols) != (spec.rows, spec.cols):
          raise ConfigurationMismatchError(
              f"expected {spec.rows}x{spec.cols}, got {config.rows}x{config.cols}"
          )
      if config.tx_elements != spec.tx_elements:
          raise ConfigurationMismatchError("transmitter element count mismatch")
      if config.frequency_hz != spec.frequency_hz:
          raise ConfigurationMismatchError("carrier frequency mismatch")
  ```

  Also implement:

  ```python
  def inventory_samples(
      workspace_root: Path,
      schema: DatasetSchemaLock,
  ) -> SampleInventory:
      return _index_every_released_sample_file(workspace_root, schema)

  def resolve_sample_paths(
      inventory: SampleInventory,
      config_id: str,
      beam_id: int,
      scene_id: str,
  ) -> tuple[Path, Path, Path]:
      return inventory.require_unique_triplet(config_id, beam_id, scene_id)
  ```

  The audit report must retain each real relative path, SHA-256, parsed field name, raw value string, and beam-setting inventory. Unknown fields are retained rather than discarded. A configuration without directly encoded rows/columns may only receive them from an explicit `shape_evidence` record containing `kind="reference_generator"`, reference revision, actual script path, script SHA-256, and explicit rows/columns; never compute them from element count.

  The schema lock must also freeze `data_root` relative to `workspace_root`, exact path-resolution rules for scene-to-height, configuration/beam-to-beam-map, and configuration/beam/scene-to-radiomap, expected source shapes, transmitter coordinate, target sentinels, archive SHA-256, both receipt hashes, and all real configuration/beam-setting hashes. Every logical lookup must match exactly one file; zero or multiple matches fail.

- [ ] **Step 5: Audit the extracted archive before writing the lock**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe prepare_multiconfig.py audit-schema --dataset-root E:\datasets\MultiConfigRadiomap --report E:\datasets\MultiConfigRadiomap\config_audit.json
  ```

  Inspect every discovered configuration and the pinned generator code. Update only the parser for formats actually present. Then run `freeze-schema`, which must refuse ambiguity and write `experiments/multiconfig_schema.json` containing exact configuration IDs, rows, columns, frequency, full released beam count, selected beam IDs/angles, source-file hashes, byte-faithful layout/path rules, expected shapes, Tx/sentinel rules, dataset/archive/receipt identities, dataset revision, reference-code revision, reference-script evidence, and audit-report hash. Do not handwrite guessed IDs or field names.

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe prepare_multiconfig.py freeze-schema --dataset-root E:\datasets\MultiConfigRadiomap --audit-report E:\datasets\MultiConfigRadiomap\config_audit.json --output experiments\multiconfig_schema.json
  ```

- [ ] **Step 6: Run unit and real-schema verification**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_schema.py -q
  D:\Anaconda3\envs\radioflow-win\python.exe prepare_multiconfig.py verify-schema --dataset-root E:\datasets\MultiConfigRadiomap --schema experiments\multiconfig_schema.json
  git diff --check
  ```

- [ ] **Step 7: Commit the parser and actual schema lock**

  ```powershell
  git add experiments/multiconfig_manifest.py experiments/multiconfig_schema.json tests/fixtures/multiconfig_config tests/test_multiconfig_schema.py prepare_multiconfig.py
  git commit -m "feat: lock Multi-config array and beam schema"
  ```

---

### Task 4: Create the permanent scene split and strict manifests

**Files:**

- Modify: `experiments/multiconfig_manifest.py`
- Modify: `prepare_multiconfig.py`
- Create: `tests/test_multiconfig_manifest.py`
- Create: `tests/test_multiconfig_real_data.py`

- [ ] **Step 1: Write deterministic split and manifest contract tests**

  Use an in-memory/synthetic inventory of `u1` through `u800`. Test exact 560/80/160 scene counts, pairwise disjointness, equality of scene sets across arrays, exact 4,480/640/1,280 sample counts, stable natural sorting before shuffling, stable JSON bytes, duplicate rejection, missing-file rejection, incorrect beam rejection, and existing-split immutability. Verify each array independently has exactly 800 unique scenes before comparing the three sets; never hide missing/duplicate scenes by taking a union or intersection.

  Use these records:

  ```python
  @dataclass(frozen=True)
  class SceneSplit:
      seed: int
      algorithm: str
      train: tuple[str, ...]
      val: tuple[str, ...]
      test: tuple[str, ...]

  @dataclass(frozen=True)
  class ManifestRecord:
      sample_key: str
      split: Literal["train", "val", "test"]
      scene_id: str
      array_name: str
      array_rows: int
      array_cols: int
      frequency_hz: int
      config_id: str
      beam_id: int
      steering_deg: float
      height_path: str
      beam_map_path: str
      radiomap_path: str
  ```

- [ ] **Step 2: Run the manifest tests and confirm missing APIs fail**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_manifest.py -q
  ```

- [ ] **Step 3: Implement permanent split and atomic JSONL writing**

  The split algorithm is exactly:

  ```python
  scene_sets = inventory.scene_ids_by_array()
  for array_name, scene_ids in scene_sets.items():
      if len(scene_ids) != 800:
          raise SplitContractError(
              f"{array_name}: expected 800 unique scenes, got {len(scene_ids)}"
          )
  if not (scene_sets["8x8"] == scene_sets["16x16"]
          == scene_sets["32x32"]):
      raise SplitContractError("scene sets differ across arrays")
  ordered = natural_sorted(scene_sets["8x8"])
  random.Random(42).shuffle(ordered)
  split = SceneSplit(
      seed=42,
      algorithm="python_random_v1",
      train=tuple(ordered[:560]),
      val=tuple(ordered[560:640]),
      test=tuple(ordered[640:800]),
  )
  ```

  Implement `load_or_create_scene_split`, `build_manifest`, `validate_manifest`, and `write_manifest_jsonl`. Build records only from the locked filesystem inventory and schema evidence; do not use the released `metadata.csv` as an index. Once the split file exists, load and validate seed 42, algorithm name, the complete 800-scene universe, 560/80/160 counts, and disjointness; never silently regenerate it. Use POSIX paths relative to `workspace_root` and stable records sorted by natural scene order then beam ID. Define `sample_key = f"{scene_id}|{array_name}|beam{beam_id:02d}"`. Canonical JSON uses sorted keys and compact separators, and writers use a temporary sibling plus `os.replace`.

  Build `visualization_cases_seed42.json` from the same sorted test scenes: fix four scene IDs deterministically and the common angles `[-28.0, 0.0, 21.0]`. The file is shared across all six models.

- [ ] **Step 4: Generate and verify all three real manifests**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe prepare_multiconfig.py build-manifests --dataset-root E:\datasets\MultiConfigRadiomap --schema experiments\multiconfig_schema.json --manifest-dir E:\datasets\MultiConfigRadiomap\manifests
  $env:MULTICONFIG_ROOT='E:\datasets\MultiConfigRadiomap'
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_real_data.py -m dataset -q
  ```

  The real-data test must assert 6,400 records per array; 4,480/640/1,280 split counts; exact beam IDs and angles; exactly 560 training, 80 validation, and 160 test samples per selected beam; matching scene sets across arrays; and the absence of path, split, or key duplication. It also verifies that the schema-bound archive, download receipt, extraction receipt, real configuration files, beam-setting files, and reference scripts still have their locked hashes.

- [ ] **Step 5: Run all data-preparation tests**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_download.py tests/test_multiconfig_schema.py tests/test_multiconfig_manifest.py -q
  git diff --check
  ```

- [ ] **Step 6: Commit manifest code, not generated data**

  ```powershell
  git add experiments/multiconfig_manifest.py prepare_multiconfig.py tests/test_multiconfig_manifest.py tests/test_multiconfig_real_data.py
  git commit -m "feat: build fixed Multi-config scene manifests"
  ```

---

### Task 5: Add the three-channel Multi-config RadioFlow dataset adapter

**Files:**

- Create: `data_loaders/multiconfig.py`
- Create: `tests/test_multiconfig_dataset.py`
- Modify: `tests/test_multiconfig_real_data.py`
- Modify: `prepare_multiconfig.py`

- [ ] **Step 1: Write preprocessing and dataset tests**

  Include tests named:

  ```text
  test_db_normalization_round_trip
  test_target_mask_excludes_floor_and_buildings
  test_unexpected_target_value_fails
  test_empty_valid_mask_fails
  test_height_max_uses_train_scenes_only
  test_continuous_maps_use_bilinear
  test_valid_mask_uses_nearest
  test_tx_mask_has_one_pixel_at_127_127
  test_dataset_returns_fixed_shapes_and_channel_order
  test_val_and_test_reuse_train_height_max
  test_np_load_rejects_object_arrays
  test_collate_preserves_metadata_as_records
  ```

  For the hand example `[-300, -299; 1000, -150]`, assert mask `[False, True; False, True]` and normalized target `[0, 1/300; 0, 0.5]`.

- [ ] **Step 2: Run the new dataset test in the red state**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_dataset.py -q
  ```

- [ ] **Step 3: Implement strict normalization, interpolation, and loading**

  Provide these functions and dataset constructor:

  ```python
  def normalize_db(values_db: Tensor, floor_db: float = -300.0,
                   ceiling_db: float = 0.0) -> Tensor:
      return ((values_db.clamp(floor_db, ceiling_db) - floor_db)
              / (ceiling_db - floor_db))

  def denormalize_db(values: Tensor, floor_db: float = -300.0,
                     ceiling_db: float = 0.0) -> Tensor:
      return values * (ceiling_db - floor_db) + floor_db

  def prepare_target(radiomap_db: Tensor) -> tuple[Tensor, Tensor]:
      finite = torch.isfinite(radiomap_db)
      valid = finite & (radiomap_db > -300.0) & (radiomap_db < 0.0)
      known = valid | (radiomap_db == -300.0) | (radiomap_db == 1000.0)
      if not bool(finite.all()) or not bool(known.all()):
          raise TargetValueError("radiomap contains non-finite or unknown values")
      if not bool(valid.any()):
          raise EmptyValidMaskError("radiomap has no valid propagation cells")
      target = torch.zeros_like(radiomap_db, dtype=torch.float32)
      target[valid] = normalize_db(radiomap_db[valid].float())
      return target, valid
  ```

  Use bilinear interpolation with `align_corners=False` for beam maps and target values, nearest-neighbor for validity masks, and clear target values again wherever the resized mask is false. Load every NPY with `np.load(path, allow_pickle=False)`. Validate source dimensions from the actual schema lock, output shapes, dtypes, finiteness, positive `height_max`, and the exact one-pixel transmitter position `(127, 127)`.

  ```python
  class MultiConfigRadiomapDataset(Dataset):
      def __init__(
          self,
          *,
          dataset_root: Path,
          manifest_path: Path,
          split: Literal["train", "val", "test"],
          schema: DatasetSchemaLock,
          height_stats: HeightStats,
          output_size: tuple[int, int] = (256, 256),
      ) -> None:
          self.records = load_and_validate_records(manifest_path, split)
          self.dataset_root = dataset_root
          self.schema = validate_schema_identity(schema, dataset_root)
          self.height_stats = validate_height_stats_identity(
              height_stats, manifest_path, split
          )
          self.height_max = require_positive_finite(height_stats.height_max)
          self.output_size = require_fixed_size(output_size, (256, 256))
  ```

  There is no non-strict mode. Resolve paths as `workspace_root / manifest_relative_path` and validate source shapes against the supplied schema. Height preprocessing rejects non-finite or negative values and computes `height.float() / train_height_max`; do not clip validation/test heights above one. Wrap NPY loading errors, including object arrays rejected by `allow_pickle=False`, in a project `DataFormatError` that contains the exact source path.

  Return `condition float32[3,256,256]`, `target float32[1,256,256]`, `valid_mask bool[1,256,256]`, and metadata with fixed keys `sample_key, split, scene_id, array_name, array_rows, array_cols, frequency_hz, config_id, beam_id, steering_deg, height_path, beam_map_path, radiomap_path, tx_rc`. Channel order is exactly Tx mask, normalized height, normalized beam map. Add a custom collate function that stacks tensors but returns metadata as `list[dict]`.

- [ ] **Step 4: Compute and freeze train-only height statistics**

  Implement `compute_train_height_max(dataset_root, train_records)` over unique training height files only. First prove the same scene ID in all three manifests resolves to the same height file. Read exactly 560 unique training-scene height files once, and never open a validation/test height file. The `compute-height-stats` command writes canonical JSON with `height_max`, `derived_from="train"`, `scene_count=560`, the complete 560-path set and hashes, split SHA-256, all three manifest SHA-256 values, and schema identity. Validation and test constructors receive this `HeightStats` object and may not recompute it.

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe prepare_multiconfig.py compute-height-stats --dataset-root E:\datasets\MultiConfigRadiomap --manifest-dir E:\datasets\MultiConfigRadiomap\manifests
  ```

- [ ] **Step 5: Run synthetic and real decoded-sample gates**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_dataset.py -q
  $env:MULTICONFIG_ROOT='E:\datasets\MultiConfigRadiomap'
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_real_data.py -m dataset -q
  git diff --check
  ```

  The real test decodes at least one record from every split and array plus all eight beams of one scene per array. It asserts fixed shapes, channel order, finite continuous tensors, boolean mask, at least one valid pixel, complete metadata keys, and correct unique path resolution for every selected beam.

- [ ] **Step 6: Commit the adapter**

  ```powershell
  git add data_loaders/multiconfig.py tests/test_multiconfig_dataset.py tests/test_multiconfig_real_data.py prepare_multiconfig.py
  git commit -m "feat: add Multi-config RadioFlow data adapter"
  ```

---

### Task 6: Implement deterministic RadioFlow CFG sampling and masked metrics

**Files:**

- Create: `evaluation/radioflow_sampling.py`
- Create: `evaluation/radiomap_metrics.py`
- Create: `tests/test_radioflow_sampling.py`
- Create: `tests/test_radiomap_metrics.py`

- [ ] **Step 1: Write deterministic sampling tests**

  With a recording fake model, assert that two-step Euler calls `forward_with_cfg` exactly at `t=0.0` and `t=0.5`, never calls ordinary `forward`, computes the condition embedding once, rejects non-positive step counts, and changes output when the fake model responds to CFG scale. Test that a canonical noise key based only on scene ID and common angle gives identical CPU noise across array sizes, model sizes, processes, and CFG candidates.

- [ ] **Step 2: Write hand-computed masked metric tests**

  Assert perfect prediction gives zero dB-RMSE, dB-MAE, and MSE, NMSE zero, SSIM one, and mathematical PSNR infinity. Assert changes outside `valid_mask` affect no metric, while any non-finite prediction in a valid cell fails. Test global pixel weighting with two samples having unequal valid counts, eight per-beam rows, normalized/dB round trips, mask erosion for SSIM, and an empty valid mask failure.

- [ ] **Step 3: Run both modules in the red state**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_radioflow_sampling.py tests/test_radiomap_metrics.py -q
  ```

- [ ] **Step 4: Implement stable noise and the locked RadioFlow Euler path**

  ```python
  def make_sample_noise(
      scene_id: str,
      steering_deg: float,
      shape: tuple[int, int, int] = (1, 256, 256),
      *,
      base_seed: int = 42,
      dtype: torch.dtype = torch.float32,
  ) -> Tensor:
      material = f"{base_seed}|{scene_id}|{steering_deg:.6f}".encode("utf-8")
      seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
      generator = torch.Generator(device="cpu").manual_seed(seed)
      return torch.randn(shape, generator=generator, dtype=dtype, device="cpu")

  @torch.inference_mode()
  def euler_cfg_sample(
      model: DiffUNet,
      condition: Tensor,
      x0: Tensor,
      *,
      cfg_scale: float,
      steps: int = 2,
      use_amp: bool = True,
  ) -> Tensor:
      if steps <= 0:
          raise ValueError("steps must be positive")
      device_type = condition.device.type
      with torch.amp.autocast(
          device_type=device_type,
          dtype=torch.float16,
          enabled=use_amp and device_type == "cuda",
      ):
          embedding = model.embed_model(condition)
          x = x0
          dt = 1.0 / steps
          for k in range(steps):
              t = torch.full((x.shape[0],), k / steps,
                             device=x.device, dtype=x.dtype)
              velocity = model.forward_with_cfg(
                  image=condition, x=x, step=t, embedding=embedding,
                  cfg_scale=cfg_scale,
              )
              x = x + dt * velocity
      return x.float()
  ```

  This deliberately uses RadioFlow's own `forward_with_cfg`, including its existing conditional/unconditional embedding behavior. Do not reimplement or “correct” its CFG equation in the benchmark adapter.

- [ ] **Step 5: Implement globally accumulated valid-region metrics**

  `MetricAccumulator.update` first rejects non-finite predictions in valid cells, records raw valid-pixel fractions below zero and above one, then clamps prediction to `[0,1]` for all accuracy metrics and visualizations. Accumulate numerators and denominators over all valid pixels rather than averaging per-image RMSE values:

  ```python
  error_norm = prediction_eval.double() - target_norm.double()
  mask = valid_mask.bool()
  sum_sq_norm += error_norm[mask].square().sum().item()
  sum_abs_db += (error_norm[mask].abs() * 300.0).sum().item()
  valid_pixels += int(mask.sum().item())
  db_rmse = math.sqrt(90_000.0 * sum_sq_norm / valid_pixels)
  db_mae = sum_abs_db / valid_pixels
  mse = sum_sq_norm / valid_pixels
  nmse = sum_sq_norm / max(sum_target_sq_norm, 1e-12)
  psnr = math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
  ```

  Compute an 11x11 SSIM map and include only locations whose complete 11x11 mask window is valid; aggregate SSIM sums and valid-window counts globally. Raise if any evaluated sample has no valid SSIM window. Maintain one accumulator per selected beam plus an independent overall accumulator; do not average the eight beam summaries to get the overall row.

  Keep `math.inf` for perfect-case PSNR in the in-memory metric API. Canonical JSON writers encode that exceptional exact-perfect value as `psnr=null` plus `psnr_infinite=true`, use `allow_nan=False`, and reject every other non-finite output; normal benchmark predictions are expected to have a finite PSNR.

- [ ] **Step 6: Run focused and earlier data tests**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_radioflow_sampling.py tests/test_radiomap_metrics.py tests/test_multiconfig_dataset.py -q
  git diff --check
  ```

- [ ] **Step 7: Commit sampling and metrics**

  ```powershell
  git add evaluation/radioflow_sampling.py evaluation/radiomap_metrics.py tests/test_radioflow_sampling.py tests/test_radiomap_metrics.py
  git commit -m "feat: add deterministic RadioFlow sampling and metrics"
  ```

---

### Task 7: Add locked training configuration, masked CFM loss, and exact accumulation

**Files:**

- Create: `training/config.py`
- Create: `training/masked_flow_loss.py`
- Create: `training/optimization.py`
- Create: `tests/test_masked_flow_loss.py`
- Create: `tests/test_gradient_accumulation.py`
- Create: `tests/test_train_config.py`

- [ ] **Step 1: Write masked-loss tests**

  For predictions `[1, 3; 100, -99]`, targets `[0, 1; 0, 0]`, and mask `[1, 1; 0, 0]`, assert loss 2.5 and zero gradients in invalid cells. Reject shape mismatch, non-boolean masks, non-finite predictions/targets in valid cells, and a zero-valid-pixel mask.

- [ ] **Step 2: Write configuration and accumulation tests**

  Assert Lite derives micro-batch 2, accumulation 8, AMP on, activation checkpointing off; Large derives micro-batch 1, accumulation 16, AMP on, activation checkpointing on; both give effective batch 16 and 280 optimizer steps per 4,480-sample epoch. With a toy linear model and unequal mask counts, prove one accumulated update matches a true effective-batch pixel-weighted loss. Assert scheduler and EMA update only after a successful optimizer step and the final short window uses its actual records and valid-pixel denominator.

- [ ] **Step 3: Run the tests in the red state**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_masked_flow_loss.py tests/test_gradient_accumulation.py tests/test_train_config.py -q
  ```

- [ ] **Step 4: Implement immutable scientific configuration**

  ```python
  @dataclass(frozen=True)
  class MultiConfigTrainConfig:
      array_size: Literal["8x8", "16x16", "32x32"]
      model_size: Literal["lite", "large"]
      dataset_root: Path
      manifest_dir: Path
      run_root: Path
      seed: int = 42
      learning_rate: float = 1e-3
      weight_decay: float = 1e-5
      warmup_ratio: float = 0.10
      ema_decay: float = 0.999
      max_epochs: int = 200
      early_stopping_patience: int = 20
      num_workers: int = 2
      resolution: int = 256
      use_amp: bool = True
      amp_dtype: str = "float16"

      @property
      def micro_batch_size(self) -> int:
          return 2 if self.model_size == "lite" else 1

      @property
      def accumulation_steps(self) -> int:
          return 8 if self.model_size == "lite" else 16

      @property
      def activation_checkpointing(self) -> bool:
          return self.model_size == "large"
  ```

  Validate every fixed field on load from JSON and reject unknown keys. A run config records derived values but the CLI cannot override them.

  Split configuration hashing into immutable scientific identity and invocation controls. Model/data/optimizer fields, the 200-epoch horizon, AMP request, and derived batch strategy enter `config_sha256`; `resume`, `stop_after_epoch`, and `smoke_optimizer_steps` are logged separately and never enter that hash, so a five-epoch pause can resume under the same identity. On CPU, record `amp_requested=True` and `scaler_enabled=False` rather than claiming AMP was active.

- [ ] **Step 5: Implement masked velocity loss and optimizer-step scheduler**

  ```python
  def masked_velocity_mse(
      predicted_velocity: Tensor,
      target_velocity: Tensor,
      valid_mask: Tensor,
  ) -> Tensor:
      _validate_velocity_tensors(predicted_velocity, target_velocity, valid_mask)
      valid_count = valid_mask.sum()
      if int(valid_count.item()) == 0:
          raise ValueError("valid_mask contains zero valid pixels")
      difference = (
          predicted_velocity.float() - target_velocity.float()
      ).masked_select(valid_mask)
      if not bool(torch.isfinite(difference).all()):
          raise ValueError("non-finite velocity in valid region")
      return difference.square().mean()
  ```

  Selecting before squaring ensures invalid-region infinities cannot create `0 * inf` gradients. Group DataLoader batches into accumulation windows before backward. Compute `total_valid` for the full window as a Python integer, then create any weighting scalar on the loss device in FP32; backpropagate each micro-batch mean multiplied by `micro_valid / total_valid`. This exactly equals one effective-batch valid-pixel objective even when masks differ. Call `optimizer.zero_grad(set_to_none=True)` only at window start and after a completed or skipped step. Accumulate reported epoch loss by total valid-pixel squared-error numerator/denominator, not by averaging micro-batch means.

  Use `ceil(len(train_loader)/accumulation_steps)` optimizer steps per epoch and a 10% warmup plus cosine schedule over `280 * 200 = 56,000` planned optimizer steps. Define the exact multiplier used for the upcoming optimizer step:

  ```python
  def lr_multiplier(step_index: int, total_steps: int, warmup_steps: int) -> float:
      optimizer_step_number = min(step_index + 1, total_steps)
      if optimizer_step_number <= warmup_steps:
          return optimizer_step_number / warmup_steps
      progress = ((optimizer_step_number - warmup_steps)
                  / (total_steps - warmup_steps))
      return 0.5 * (1.0 + math.cos(math.pi * progress))
  ```

  Test the learning rate actually used at optimizer steps 1, 5,600, and 56,000; overflow skips must not advance it, and resumed next-step LR must equal uninterrupted training.

  Use `torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")` and `torch.amp.GradScaler("cuda", enabled=device.type == "cuda")`. Compare pre/post scaler values to detect skipped overflow steps; update the reused `train.ModelEMA`, scheduler, and optimizer-step counter only when the optimizer ran.

- [ ] **Step 6: Verify all training primitives**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_masked_flow_loss.py tests/test_gradient_accumulation.py tests/test_train_config.py -q
  git diff --check
  ```

- [ ] **Step 7: Commit training primitives**

  ```powershell
  git add training/config.py training/masked_flow_loss.py training/optimization.py tests/test_masked_flow_loss.py tests/test_gradient_accumulation.py tests/test_train_config.py
  git commit -m "feat: add masked RadioFlow training primitives"
  ```

---

### Task 8: Implement atomic checkpoints and exact epoch-boundary resume

**Files:**

- Create: `training/checkpointing.py`
- Create: `tests/test_checkpoint_resume.py`

- [ ] **Step 1: Write continuous-versus-resumed training tests**

  Compare two deterministic optimizer steps with one step, atomic save, fresh model/EMA/optimizer/scheduler/scaler objects, strict load, then the second step. Assert equality of model weights, EMA weights, optimizer tensors, scheduler state, counters, history, and the next values from Python, NumPy, CPU Torch, CUDA Torch when available, and the dedicated DataLoader generator. Also test missing files, missing keys, a truncated file, wrong array/model/config/manifest/split/schema/archive hashes, wrong state keys, and a non-finite best metric.

- [ ] **Step 2: Run the checkpoint test in the red state**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_checkpoint_resume.py -q
  ```

- [ ] **Step 3: Implement the strict checkpoint schema**

  ```python
  @dataclass(frozen=True)
  class CheckpointIdentity:
      array_size: str
      model_size: str
      condition_channels: int
      parameter_count: int
      manifest_sha256: str
      split_sha256: str
      schema_sha256: str
      config_sha256: str
      archive_sha256: str
      dataset_revision: str
      radioflow_upstream_base: str
      git_commit: str
      seed: int
  ```

  The top-level checkpoint keys are exactly `schema_version`, `model`, `ema`, `optimizer`, `scheduler`, `scaler`, `trainer_state`, `rng_state`, and `run_identity`. `trainer_state` stores `completed_epochs`, `next_epoch_index`, `optimizer_step`, `micro_batches_seen`, `samples_seen`, `best_val_db_rmse`, `epochs_without_improvement`, and complete epoch history. `rng_state` stores Python, NumPy, Torch CPU, all CUDA device states, and the DataLoader generator state.

  For auditability, use unambiguous counters `completed_epochs`, zero-based `next_epoch_index`, `optimizer_step`, `micro_batches_seen`, and `samples_seen`. Thus `next_epoch_index=5` means five epochs are complete and the next human-readable epoch is 6.

  Save to a temporary sibling, call `flush()` and `os.fsync()`, close it, then use `os.replace`. Load only locally produced full-state checkpoints with `torch.load(path, map_location="cpu", weights_only=False)`, validate every required field and identity value before mutating live objects, then call `load_state_dict(..., strict=True)` for model and EMA. The resume contract is the beginning of `next_epoch_index`; do not claim mid-DataLoader recovery. Rebuild `metrics.csv` from checkpoint history so resume cannot duplicate rows.

  Implement a separate `load_ema_for_evaluation` that loads only the strict EMA state after all identity, hash, and parameter-count checks. It raises on every error and contains no exception-swallowing branch.

- [ ] **Step 4: Run checkpoint and prior training tests**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_checkpoint_resume.py tests/test_gradient_accumulation.py tests/test_masked_flow_loss.py -q
  git diff --check
  ```

- [ ] **Step 5: Commit checkpoint support**

  ```powershell
  git add training/checkpointing.py tests/test_checkpoint_resume.py
  git commit -m "feat: add strict RadioFlow checkpoint resume"
  ```

---

### Task 9: Add execution-only activation checkpointing and a locked model factory

**Files:**

- Modify: `model/model.py`
- Modify: `model/unet/basic_unet.py`
- Modify: `model/unet/basic_unet_denose.py`
- Create: `training/model_factory.py`
- Create: `tests/test_activation_checkpointing.py`
- Modify: `tests/test_radioflow_framework_lock.py`

- [ ] **Step 1: Write architecture-invariance tests first**

  Instantiate identical Lite models with checkpointing off/on, copy strict state, set `cfg_drop_prob=0`, and compare forward outputs and parameter gradients on a small valid spatial input. Assert identical parameter counts and exact `state_dict().keys()`. Spy on `torch.utils.checkpoint.checkpoint` to prove calls occur only during training with gradients enabled, never during eval or inference. Instantiate Large for count/state-key checks without a CPU full-resolution forward.

- [ ] **Step 2: Confirm the constructor switch is initially missing**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_activation_checkpointing.py tests/test_radioflow_framework_lock.py -q
  ```

- [ ] **Step 3: Add non-parameter execution switches**

  Add `activation_checkpointing: bool = False` to `DiffUNet`, `BasicUNetEncoder`, and `BasicUNetDe`, and pass it through. Store only a Python boolean. In both encoder and decoder use:

  ```python
  def _run_block(self, module: nn.Module, *args: Tensor) -> Tensor:
      if (self.activation_checkpointing and self.training
              and torch.is_grad_enabled()):
          return checkpoint(
              module, *args, use_reentrant=False, preserve_rng_state=True
          )
      return module(*args)
  ```

  Route encoder `conv_0` and `down_1` through `down_4` through the helper. Route decoder `conv_0`, each used cross-attention block, `down_1` through `down_4`, and `upcat_4` through `upcat_1` through it. Do not checkpoint `final_conv` or the timestep MLP. Do not add a layer, parameter, buffer, or state key.

- [ ] **Step 4: Implement the only benchmark model factory**

  ```python
  EXPECTED_PARAMETER_COUNTS = {"lite": 3_994_859, "large": 54_126_059}

  def build_locked_radioflow(
      model_size: Literal["lite", "large"],
  ) -> DiffUNet:
      activation_checkpointing = model_size == "large"
      network = DiffUNet(
          con_channels=3,
          model_size=model_size,
          activation_checkpointing=activation_checkpointing,
      )
      if type(network.embed_model) is not BasicUNetEncoder:
          raise FrameworkLockError("unexpected condition encoder")
      if type(network.model) is not BasicUNetDe:
          raise FrameworkLockError("unexpected velocity decoder")
      if network.cfg_drop_prob != 0.25:
          raise FrameworkLockError("RadioFlow CFG dropout must remain 0.25")
      actual = sum(parameter.numel() for parameter in network.parameters())
      if actual != EXPECTED_PARAMETER_COUNTS[model_size]:
          raise FrameworkLockError(
              f"parameter count changed: expected "
              f"{EXPECTED_PARAMETER_COUNTS[model_size]}, got {actual}"
          )
      return network
  ```

  Train and evaluation entry points must call this factory; no injectable alternate model class or checkpoint-policy argument is allowed. The activation-checkpoint equivalence test constructs `DiffUNet` directly so it can exercise both execution modes without weakening the production factory.

- [ ] **Step 5: Run architecture, sampling, and loss regressions**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_activation_checkpointing.py tests/test_radioflow_framework_lock.py tests/test_radioflow_sampling.py tests/test_masked_flow_loss.py -q
  git diff --check
  ```

- [ ] **Step 6: Commit the execution-only change**

  ```powershell
  git add model/model.py model/unet/basic_unet.py model/unet/basic_unet_denose.py training/model_factory.py tests/test_activation_checkpointing.py tests/test_radioflow_framework_lock.py
  git commit -m "feat: checkpoint locked RadioFlow execution"
  ```

---

### Task 10: Build the resumable Multi-config RadioFlow trainer and CLI

**Files:**

- Create: `training/multiconfig_trainer.py`
- Create: `training/hardware_evidence.py`
- Create: `train_multiconfig.py`
- Create: `tests/test_train_multiconfig_cli.py`
- Create: `tests/test_multiconfig_train_integration.py`
- Create: `tests/test_hardware_evidence.py`

- [ ] **Step 1: Write CLI and synthetic integration tests**

  Assert three arrays and two model sizes produce the fixed derived config and expected run path. Reject resolution, batch, accumulation, activation-checkpoint policy, CFG dropout, frequency, beam, seed, optimizer, and epoch overrides. Test `--resume none`, `--resume auto`, an explicit checkpoint, `--stop-after-epoch 5`, and `--smoke-optimizer-steps 1`. A synthetic dataset integration test must complete one optimizer step, EMA update, validation generation, `last.pt`, strict fresh-object resume, and one more step. Add a mocked `torch.cuda.OutOfMemoryError` test that writes a hashable global Large hardware-gate artifact without creating a production checkpoint; unrelated exceptions must propagate without being mislabeled OOM.

- [ ] **Step 2: Run trainer tests in the red state**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_train_multiconfig_cli.py tests/test_multiconfig_train_integration.py tests/test_hardware_evidence.py -q
  ```

- [ ] **Step 3: Implement trainer initialization and deterministic loaders**

  ```python
  class MultiConfigSRMTrainer:
      def __init__(
          self,
          cfg: MultiConfigTrainConfig,
          model: DiffUNet,
          train_loader: DataLoader,
          val_loader: DataLoader,
          device: torch.device,
          train_generator: torch.Generator,
          identity: CheckpointIdentity,
      ) -> None:
          self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
          self.optimizer = torch.optim.AdamW(
              model.parameters(), lr=1e-3, weight_decay=1e-5
          )
          self.ema = ModelEMA(model, decay=0.999)
          self.scheduler = build_optimizer_step_scheduler(
              self.optimizer, total_steps=56_000, warmup_steps=5_600
          )
  ```

  Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before any CUDA context is created. The CLI must then call `seed_everything(42)` before model, DataLoader, optimizer, or EMA construction; this deliberately overrides `model/model.py`'s import-time seed 123 before parameter initialization. Set `torch.backends.cudnn.benchmark=False`, `torch.backends.cudnn.deterministic=True`, seed Python, NumPy, Torch CPU/CUDA, and the DataLoader generator, and enable deterministic algorithms.

  The construction order is fixed:

  ```python
  seed_everything(42)
  model = build_locked_radioflow(cfg.model_size).to(device)
  train_loader, val_loader, train_generator = build_loaders(cfg)
  trainer = MultiConfigSRMTrainer(
      cfg, model, train_loader, val_loader, device,
      train_generator, identity,
  )
  ```

  `MultiConfigSRMTrainer` asserts every model parameter is already on `device` before `ModelEMA` deep-copies it, keeping model and EMA colocated. Use `shuffle=True` only for train; validation is manifest order. Set workers to 2, use the custom collate function, and define worker seeds from `torch.initial_seed()`. Fail if a used operation is nondeterministic rather than silently changing seeds.

- [ ] **Step 4: Implement CFM training, generation-based validation, and recovery**

  Preserve RadioFlow CFM exactly:

  ```python
  x0 = torch.randn_like(target)
  t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, target)
  vt = self.model(image=condition, x=xt, pred_type="denoise", step=t)
  micro_loss = masked_velocity_mse(vt, ut, valid_mask)
  weighted_loss = micro_loss * (valid_mask.sum() / window_valid_count)
  self.scaler.scale(weighted_loss).backward()
  ```

  Start each accumulation window with `optimizer.zero_grad(set_to_none=True)`. After scaler step/update, advance EMA, scheduler, and optimizer counter only if the optimizer ran, then clear gradients with `set_to_none=True`. Record cumulative samples and micro-batches even when an overflow skips the optimizer update.

  At every completed epoch, evaluate all 640 validation samples using the EMA model, two-step Euler, CFG 1.0, and per-scene/per-angle deterministic noise from Task 6. Save `best.pt` only when valid-region dB-RMSE strictly improves; increment patience otherwise. Save `last.pt` after every complete epoch. Early-stop after 20 consecutive non-improving epochs.

  `--stop-after-epoch 5` pauses after writing an epoch-boundary `last.pt`; it does not change `max_epochs=200`, total scheduler steps, or declare training complete. `--resume auto` resumes only an identity-compatible `last.pt`. Restore `metrics.csv` from checkpoint history.

- [ ] **Step 5: Implement Large OOM evidence as a benchmark terminal state**

  Catch only `torch.cuda.OutOfMemoryError` around the approved Large smoke optimizer step. Atomically write `run_root/_hardware/large_hardware_gate.json` containing the original exception, GPU name/total memory, PyTorch/CUDA versions, all data/config/framework hashes, model size and exact parameter count, condition/output shapes shared by all arrays, 256 resolution, AMP dtype, micro-batch 1, accumulation 16, activation checkpointing on, and peak allocated bytes if available. Include `scope=["8x8/large","16x16/large","32x32/large"]` and explain that array identity changes condition values but not tensor or network shape. The orchestration script validates this evidence hash, skips all Large runs, and continues every Lite run. A Lite OOM or any non-OOM failure remains fatal.

- [ ] **Step 6: Make the CLI fail closed and record immutable config**

  Required arguments are dataset root, manifest directory, array, model size, run root, device, and resume mode. Before model allocation, verify provenance, schema/split/manifest/height hashes, exact sample counts, and one decoded sample per split. Write canonical `config.json` containing all fixed and derived values and its own hash input. Never overwrite a non-matching run directory.

  `--smoke-optimizer-steps` is an execution gate, not a training run: it must derive an isolated path `run_root/_smoke/<array>/<model_size>`, may replace only its own prior smoke artifacts, and must never create or mutate production `best.pt`, `last.pt`, `config.json`, or history.

- [ ] **Step 7: Run trainer integration and the full CPU suite**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_train_multiconfig_cli.py tests/test_multiconfig_train_integration.py tests/test_hardware_evidence.py -q
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -m "not dataset and not gpu and not slow" -q
  git diff --check
  ```

- [ ] **Step 8: Commit the trainer**

  ```powershell
  git add training/multiconfig_trainer.py training/hardware_evidence.py train_multiconfig.py tests/test_train_multiconfig_cli.py tests/test_multiconfig_train_integration.py tests/test_hardware_evidence.py
  git commit -m "feat: train Multi-config SRM with RadioFlow"
  ```

---

### Task 11: Build CFG selection, strict test evaluation, runtime measurement, and reporting

**Files:**

- Create: `evaluation/runtime_benchmark.py`
- Create: `evaluation/visualization.py`
- Create: `evaluation/multiconfig_evaluator.py`
- Create: `evaluate_multiconfig.py`
- Create: `tests/test_cfg_selection.py`
- Create: `tests/test_runtime_benchmark.py`
- Create: `tests/test_multiconfig_evaluate_cli.py`
- Create: `tests/test_multiconfig_evaluate_integration.py`

- [ ] **Step 1: Write CFG-selection and fail-closed CLI tests**

  Test the fixed candidates `[1.0, 1.5, 2.0, 2.5]`, minimum validation dB-RMSE selection, smaller-scale tie breaking, and `cfg_selection.json` hashes. `select-cfg` writes the immutable source selection under the training run directory; if it already exists, the command may only validate it and exit successfully, never overwrite it. If a completed test receipt already exists, selection changes are forbidden. Test that `test` requires this file, rejects a changed checkpoint/config/manifest/split/schema/selection hash, has no CFG/solver/step/resolution/batch override, refuses an already completed result directory, and creates no final metrics after a loading failure.

- [ ] **Step 2: Write metrics-output, runtime, and visualization tests**

  Assert the test accumulator receives exactly 1,280 samples and each selected beam exactly 160. Assert `metrics_per_beam.csv` has exactly eight rows with fixed columns. With mocked CUDA timing, require 20 warmups, 100 measured calls, synchronization boundaries, p50 and p95 calculation, and `max_memory_allocated`. Assert fixed visualization cases and stable filenames, fixed GT/prediction range `[-300,0]` dB, fixed absolute-error range `[0,300]` dB, gray invalid cells, and compressed NPZ values containing prediction, target, mask, and metadata. Inject failures after prediction, metrics, runtime, and visualization stages and assert no final result directory or completion receipt is published.

- [ ] **Step 3: Run evaluation tests in the red state**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_cfg_selection.py tests/test_runtime_benchmark.py tests/test_multiconfig_evaluate_cli.py tests/test_multiconfig_evaluate_integration.py -q
  ```

- [ ] **Step 4: Implement the two-stage validation protocol**

  `select-cfg` strictly loads the EMA state from `best.pt`, then evaluates all 640 validation samples four times with identical deterministic initial noise. Write canonical `runs/.../cfg_selection.json` containing all four metric dictionaries, selected epoch, best validation dB-RMSE at CFG 1.0, selected scale, selected-scale validation dB-RMSE, tie-break rule, checkpoint SHA-256, and all identity hashes. The selected epoch must equal the best-checkpoint epoch. If this file already exists, validate its bytes/hashes and do no generation; if final `run_manifest.json(status="complete")` exists, reject any attempt to alter selection. The test subcommand may only consume this frozen selection.

  This makes model selection precise and affordable: epoch selection is EMA/two-step-Euler/CFG-1.0 validation during training; CFG selection scans the frozen best epoch once after training. Test labels are not read during either choice.

- [ ] **Step 5: Implement one-time test evaluation and per-beam output**

  Fix `split=test`, batch 1, EMA on, solver Euler, steps 2, resolution 256. Treat `run_manifest.json` with `status="complete"` as the sole completion marker and refuse to start when it exists. The final result directory must otherwise be absent. Create a sibling `<model>.staging-<uuid>` directory, copy the frozen selection into it, and perform predictions, all metrics/CSV/NPZ writes, runtime benchmark, and fixed-case visualization there. Verify all 1,280 records, eight 160-sample beam groups, files, and hashes; write `run_manifest.json` last inside staging; then atomically rename the complete staging directory to the final result directory. On failure, no final result directory, `metrics_test.json`, or completion receipt may be published.

  `metrics_test.json` records at least schema version, array/model size, sample and valid-pixel counts, selected epoch, best validation dB-RMSE at CFG 1.0, chosen CFG, chosen-CFG validation dB-RMSE, solver, Euler steps, EMA flag, CFG-selection SHA-256, all other hashes, dB-RMSE, dB-MAE, MSE, NMSE, PSNR, SSIM, and raw prediction fractions below zero/above one. `run_manifest.json` repeats the selection hash and hashes every published artifact. `metrics_per_beam.csv` columns are `angle_deg,beam_id,n_samples,n_valid_pixels,db_rmse,db_mae,mse,nmse,psnr,ssim`. Overall values come from the global accumulator, never the mean of rows.

- [ ] **Step 6: Implement efficiency measurement and common visualizations**

  The `test` transaction calls `benchmark_generation` after accuracy prediction and before visualization. It includes condition encoding and both conditional/unconditional branches at each of two Euler steps, but excludes loading, DataLoader, metrics, and image writes. Use batch 1 and AMP matching evaluation. Call `model.eval()`, `torch.cuda.empty_cache()`, reset peak statistics, run 20 warmups, reset peak statistics again, then perform 100 synchronized timed runs and read `torch.cuda.max_memory_allocated(device)`. Report p50, p95, mean, standard deviation, GPU name/memory, PyTorch/CUDA versions, parameter count, and actual `best.pt` byte size. Peak inference allocation includes loaded model plus generation activations. Every formal training run records its own peak training allocation in `training_runtime.json`; smoke memory is only a feasibility gate, and summary prefers the formal run value.

  Still inside the same test transaction, render each fixed case as `Height | Beam map | Ground truth dB | Prediction dB | Absolute error dB`. Use fixed GT/prediction limits `[-300,0]` dB and error limits `[0,300]` dB so independently executed models remain visually comparable. Use gray invalid pixels and stable titles containing scene, array, beam, angle, model size, and CFG.

- [ ] **Step 7: Implement six-run summary validation**

  The `summarize` subcommand accepts exactly two terminal states: six completed array/model pairs, or all three Lite pairs completed plus one valid global `large_hardware_gate.json` whose scope and identities cover all three missing Large pairs. It validates every current selection SHA-256 against the copy frozen into test metrics and the completion manifest. It rejects partial/unexplained runs and inconsistent split, schema, source revision, common angles, resolution, solver, Euler step count, EMA flag, record counts, or checkpoint/config hashes. Cross-array per-beam joins use `angle_deg`, never differing beam IDs. A completed benchmark writes a six-row Markdown and CSV table covering accuracy, selected epoch/CFG validation values, and efficiency; a hardware-blocked benchmark writes three metric rows plus three clearly labeled blocked rows referencing the same global evidence SHA-256 rather than fabricated metrics.

- [ ] **Step 8: Run all evaluation and CPU integration tests**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_cfg_selection.py tests/test_runtime_benchmark.py tests/test_multiconfig_evaluate_cli.py tests/test_multiconfig_evaluate_integration.py tests/test_radioflow_sampling.py tests/test_radiomap_metrics.py -q
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -m "not dataset and not gpu and not slow" -q
  git diff --check
  ```

- [ ] **Step 9: Commit evaluation and reporting**

  ```powershell
  git add evaluation/runtime_benchmark.py evaluation/visualization.py evaluation/multiconfig_evaluator.py evaluate_multiconfig.py tests/test_cfg_selection.py tests/test_runtime_benchmark.py tests/test_multiconfig_evaluate_cli.py tests/test_multiconfig_evaluate_integration.py
  git commit -m "feat: evaluate Multi-config RadioFlow benchmark"
  ```

---

### Task 12: Pass real-data gates and staged GPU smoke/pilot runs

**Files:**

- Create: `scripts/run_multiconfig_benchmark.ps1`
- Create: `tests/test_benchmark_script.py`
- Create: `tests/test_multiconfig_gpu_smoke.py`
- Modify only if a discovered real-format bug is proven by a fixture: the responsible data, training, or evaluation module and its focused test

- [ ] **Step 1: Write a static orchestration-script test**

  Assert the script enumerates exactly `8x8,16x16,32x32` crossed with `lite,large`, uses the fixed roots and environment interpreter, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, runs Lite pilots before full Lite, runs Large smoke before any Large pilot, and never passes scientific overrides. It stops on every Lite failure or unexpected exception. A validated Large OOM gate sets `large_blocked=true`, skips all Large work, and continues every Lite run.

- [ ] **Step 2: Run all CPU and real-data gates**

  ```powershell
  $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -m "not dataset and not gpu and not slow" -q
  $env:MULTICONFIG_ROOT='E:\datasets\MultiConfigRadiomap'
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_real_data.py -m dataset -q
  ```

  If real data exposes a parser or shape discrepancy, first add a minimal failing fixture/test that reproduces the actual format, then fix the narrow adapter. Never loosen a count, geometry, sentinel, or checksum assertion just to pass.

- [ ] **Step 3: Run one complete Lite optimizer-step smoke**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe train_multiconfig.py --dataset-root E:\datasets\MultiConfigRadiomap --manifest-dir E:\datasets\MultiConfigRadiomap\manifests --array 8x8 --model-size lite --run-root E:\RadioFlow\runs\srm_6.7ghz_common8 --device cuda:0 --resume none --smoke-optimizer-steps 1
  ```

  Verify 8 Lite micro-batches form one optimizer step, AMP is enabled, peak allocated training VRAM is recorded, loss is finite, EMA/scheduler advance once, and a smoke checkpoint strictly reloads.

  Confirm all smoke artifacts are under `E:\RadioFlow\runs\srm_6.7ghz_common8\_smoke\8x8\lite` and the production 8x8/Lite directory is untouched.

- [ ] **Step 4: Run one complete Large optimizer-step smoke**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe train_multiconfig.py --dataset-root E:\datasets\MultiConfigRadiomap --manifest-dir E:\datasets\MultiConfigRadiomap\manifests --array 8x8 --model-size large --run-root E:\RadioFlow\runs\srm_6.7ghz_common8 --device cuda:0 --resume none --smoke-optimizer-steps 1
  ```

  Verify 16 Large micro-batches, micro-batch 1, full 256x256 tensors, AMP, activation checkpointing, exact 54,126,059 parameters, and recorded peak allocation. If CUDA OOM persists, validate the atomic global `E:\RadioFlow\runs\srm_6.7ghz_common8\_hardware\large_hardware_gate.json` from Task 10, mark every Large pair hardware-blocked, and continue Lite work. Stop all Large work without narrowing the model or resolution. A missing/invalid gate or a non-OOM failure is fatal.

- [ ] **Step 5: Run the three five-epoch Lite pilots and resume checks**

  ```powershell
  foreach ($array in @('8x8','16x16','32x32')) {
    D:\Anaconda3\envs\radioflow-win\python.exe train_multiconfig.py --dataset-root E:\datasets\MultiConfigRadiomap --manifest-dir E:\datasets\MultiConfigRadiomap\manifests --array $array --model-size lite --run-root E:\RadioFlow\runs\srm_6.7ghz_common8 --device cuda:0 --resume auto --stop-after-epoch 5
    if ($LASTEXITCODE -ne 0) { throw "Lite pilot failed for $array" }
  }
  ```

  For each pilot, inspect five finite training/validation rows, the unchanged 200-epoch scheduler horizon, `last.pt` identity, best selection by generated validation dB-RMSE, and a no-op/resume probe that starts at epoch 6 without duplicate history.

- [ ] **Step 6: Run a Large five-epoch pilot only after the memory gate**

  Use the same command for 8x8 Large with `--stop-after-epoch 5` only when the global hardware gate is absent. Confirm resume and finite validation. If it fails for non-memory reasons, reproduce with a focused test before changing code.

- [ ] **Step 7: Commit orchestration code and any test-backed adapter correction**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -m "not dataset and not gpu and not slow" -q
  $env:MULTICONFIG_ROOT='E:\datasets\MultiConfigRadiomap'
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_real_data.py -m dataset -q
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_gpu_smoke.py -m gpu -q
  git diff --check
  git add scripts/run_multiconfig_benchmark.ps1 tests/test_benchmark_script.py tests/test_multiconfig_gpu_smoke.py
  git commit -m "chore: stage Multi-config RadioFlow benchmark"
  ```

  Add only additional files that were changed through a documented red-green correction. Do not commit run artifacts.

---

### Task 13: Train all approved runs, evaluate test once, and produce the comparison report

**Files:**

- Generated, ignored: `runs/srm_6.7ghz_common8/`
- Generated, ignored: `results/srm_6.7ghz_common8/`
- Generated, ignored: `E:\datasets\MultiConfigRadiomap\manifests/`
- No source-code change is expected in this task

- [ ] **Step 1: Finish the three Lite runs under the common stopping rule**

  ```powershell
  foreach ($array in @('8x8','16x16','32x32')) {
    D:\Anaconda3\envs\radioflow-win\python.exe train_multiconfig.py --dataset-root E:\datasets\MultiConfigRadiomap --manifest-dir E:\datasets\MultiConfigRadiomap\manifests --array $array --model-size lite --run-root E:\RadioFlow\runs\srm_6.7ghz_common8 --device cuda:0 --resume auto
    if ($LASTEXITCODE -ne 0) { throw "Lite training failed for $array" }
  }
  ```

  Each run ends at epoch 200 or patience 20 and has strict `best.pt`, `last.pt`, `config.json`, `metrics.csv`, and training runtime evidence.

- [ ] **Step 2: Finish the three Large runs if the approved memory gate passed**

  If the validated global `large_hardware_gate.json` exists, do not start these runs and carry its reproducible blocked status into all three Large report rows. Otherwise execute sequentially on the single GPU:

  ```powershell
  foreach ($array in @('8x8','16x16','32x32')) {
    D:\Anaconda3\envs\radioflow-win\python.exe train_multiconfig.py --dataset-root E:\datasets\MultiConfigRadiomap --manifest-dir E:\datasets\MultiConfigRadiomap\manifests --array $array --model-size large --run-root E:\RadioFlow\runs\srm_6.7ghz_common8 --device cuda:0 --resume auto
    if ($LASTEXITCODE -ne 0) { throw "Large training failed for $array" }
  }
  ```

- [ ] **Step 3: Freeze CFG separately for every completed run**

  ```powershell
  foreach ($array in @('8x8','16x16','32x32')) {
    foreach ($size in @('lite','large')) {
      $best = "E:\RadioFlow\runs\srm_6.7ghz_common8\$array\$size\best.pt"
      if (Test-Path -LiteralPath $best) {
        D:\Anaconda3\envs\radioflow-win\python.exe evaluate_multiconfig.py select-cfg --dataset-root E:\datasets\MultiConfigRadiomap --manifest-dir E:\datasets\MultiConfigRadiomap\manifests --array $array --model-size $size --run-root E:\RadioFlow\runs\srm_6.7ghz_common8 --results-root E:\RadioFlow\results\srm_6.7ghz_common8 --device cuda:0
        if ($LASTEXITCODE -ne 0) { throw "CFG selection failed for $array/$size" }
      } elseif ($size -ne 'large' -or -not (Test-Path -LiteralPath 'E:\RadioFlow\runs\srm_6.7ghz_common8\_hardware\large_hardware_gate.json')) {
        throw "missing completed or hardware-blocked run for $array/$size"
      }
    }
  }
  ```

  Skip only runs formally marked hardware-blocked. Check that every candidate used the same 640 validation records and deterministic noise and that the chosen value belongs to the fixed grid.

- [ ] **Step 4: Evaluate each frozen test set exactly once**

  ```powershell
  foreach ($array in @('8x8','16x16','32x32')) {
    foreach ($size in @('lite','large')) {
      $selection = "E:\RadioFlow\runs\srm_6.7ghz_common8\$array\$size\cfg_selection.json"
      if (Test-Path -LiteralPath $selection) {
        D:\Anaconda3\envs\radioflow-win\python.exe evaluate_multiconfig.py test --dataset-root E:\datasets\MultiConfigRadiomap --manifest-dir E:\datasets\MultiConfigRadiomap\manifests --array $array --model-size $size --run-root E:\RadioFlow\runs\srm_6.7ghz_common8 --results-root E:\RadioFlow\results\srm_6.7ghz_common8 --device cuda:0
        if ($LASTEXITCODE -ne 0) { throw "test evaluation failed for $array/$size" }
      } elseif ($size -ne 'large' -or -not (Test-Path -LiteralPath 'E:\RadioFlow\runs\srm_6.7ghz_common8\_hardware\large_hardware_gate.json')) {
        throw "missing frozen selection or hardware block for $array/$size"
      }
    }
  }
  ```

  Do not inspect test metrics until each completed model's test command has finished. The command refuses reruns after atomically publishing `run_manifest.json(status="complete")`.

- [ ] **Step 5: Validate artifacts and create the final six-row report**

  ```powershell
  D:\Anaconda3\envs\radioflow-win\python.exe evaluate_multiconfig.py summarize --results-root E:\RadioFlow\results\srm_6.7ghz_common8 --run-root E:\RadioFlow\runs\srm_6.7ghz_common8
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest -m "not dataset and not gpu and not slow" -q
  $env:MULTICONFIG_ROOT='E:\datasets\MultiConfigRadiomap'
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_real_data.py -m dataset -q
  D:\Anaconda3\envs\radioflow-win\python.exe -m pytest tests/test_multiconfig_gpu_smoke.py -m gpu -q
  git status --short
  ```

  The GPU smoke test treats an identity-valid global Large hardware gate as the expected audited terminal state and does not deliberately trigger the same OOM again.

  The final handoff reports overall and per-angle accuracy, Lite/Large parameter counts, checkpoint size, peak training/inference VRAM, p50/p95 latency, selected epoch and CFG, exact hashes/revisions, and any Large hardware-blocked evidence. Confirm source status contains no unintended tracked changes and still only preserves the user's unrelated untracked files.

---

## Acceptance Checklist

- [ ] All unit, integration, real-data, and feasible GPU tests pass.
- [ ] Three manifests each contain exactly 6,400 records and share one permanent 560/80/160 scene split.
- [ ] Every condition is `[Tx mask, normalized height, normalized beam map]`; targets/masks are valid-region aware at 256x256.
- [ ] Both model sizes are the locked Hxxxz0/RadioFlow `DiffUNet` family with exact feature tuples and parameter counts.
- [ ] Training uses `ConditionalFlowMatcher`, masked velocity MSE, effective batch 16, AMP, original EMA behavior, and strict full-state resume.
- [ ] Large uses only the approved execution-level checkpointing strategy or is reported hardware-blocked without a scientific configuration change.
- [ ] Validation selects best epoch at CFG 1.0, then selects CFG from the fixed four-value grid using the frozen best EMA checkpoint.
- [ ] Test uses only frozen choices, deterministic common scene/angle noise, RadioFlow `forward_with_cfg`, and two Euler steps.
- [ ] Overall and per-beam metrics exclude invalid/building cells, and evaluation never uses random fallback weights.
- [ ] The final comparison has six completed rows, or three Lite rows plus explicit reproducible Large hardware-blocked evidence.
