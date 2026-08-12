$ErrorActionPreference = "Stop"

$pythonExe = "D:\Anaconda3\envs\radioflow-win\python.exe"
$datasetRoot = "E:\datasets\MultiConfigRadiomap"
$manifestRoot = "E:\datasets\MultiConfigRadiomap\manifests"
$heightStats = Join-Path $manifestRoot "height_stats_train.json"
$repoRoot = "E:\RadioFlow-worktrees\multiconfig-srm-01x"
$runRoot = "E:\RadioFlow\runs\sparse_srm_6.7ghz_5pct_single_beam"
$variant = "beam_masked"
$arrays = @("8x8", "16x16", "32x32")
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

foreach ($arraySize in $arrays) {
    $existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object {
            $_.CommandLine -like "*train_sparse_same_frequency.py*" -and
            $_.CommandLine -like ("*--array-size " + $arraySize + "*") -and
            $_.CommandLine -like ("*--variant " + $variant + "*")
        }
    if ($existing) {
        throw "A sparse same-frequency process for $variant / $arraySize already exists."
    }
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $freeMemory = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0
    if ([int]($freeMemory | Select-Object -First 1) -lt 12000) {
        throw "GPU 0 free memory is below the conservative parallel launch threshold."
    }
}

foreach ($arraySize in $arrays) {
    $manifest = Join-Path $manifestRoot ("manifest_samefreq_6.7ghz_{0}_0deg.jsonl" -f $arraySize)
    $variantRunRoot = Join-Path $runRoot $variant
    $stdout = Join-Path $runRoot ("sparse_{0}_{1}_stdout.log" -f $variant, $arraySize)
    $stderr = Join-Path $runRoot ("sparse_{0}_{1}_stderr.log" -f $variant, $arraySize)
    $arguments = @(
        "-u",
        (Join-Path $repoRoot "train_sparse_same_frequency.py"),
        "--dataset-root", $datasetRoot,
        "--manifest-path", $manifest,
        "--height-stats-path", $heightStats,
        "--run-root", $variantRunRoot,
        "--array-size", $arraySize,
        "--variant", $variant,
        "--device", "cuda:0",
        "--resume", "auto"
    )
    $process = Start-Process -FilePath $pythonExe `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    Write-Output ("started sparse {0} / {1}: PID {2}" -f $variant, $arraySize, $process.Id)
}
