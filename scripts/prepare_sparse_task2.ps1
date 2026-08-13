param(
    [string]$DatasetRoot = 'E:\datasets\MultiConfigRadiomap',
    [string]$ManifestRoot = 'E:\datasets\MultiConfigRadiomap\manifests',
    [string[]]$ArraySize = @('8x8', '16x16', '32x32')
)

$ErrorActionPreference = 'Stop'
$splitPath = Join-Path $ManifestRoot 'scene_split_seed42.json'

foreach ($array in $ArraySize) {
    if ($array -notin @('8x8', '16x16', '32x32')) {
        throw "Unsupported array size: $array"
    }
    $output = Join-Path $ManifestRoot ("manifest_singlebeam_feature5_samples819_{0}.jsonl" -f $array)
    python -c "from pathlib import Path; from experiments.sparse_task2_manifest import build_singlebeam_task2_manifest; import json; print(json.dumps(build_singlebeam_task2_manifest(dataset_root=Path(r'$DatasetRoot'), split_path=Path(r'$splitPath'), array_size='$array', output_path=Path(r'$output')), sort_keys=True))"
}
