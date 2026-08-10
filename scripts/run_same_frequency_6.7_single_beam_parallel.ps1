$ErrorActionPreference = "Stop"

$pythonExe = "D:\Anaconda3\envs\radioflow-win\python.exe"
$datasetRoot = "E:\datasets\MultiConfigRadiomap"
$manifestRoot = "E:\datasets\MultiConfigRadiomap\manifests"
$heightStats = Join-Path $manifestRoot "height_stats_train.json"
$repoRoot = "E:\RadioFlow-worktrees\multiconfig-srm-01x"
$runRoot = "E:\RadioFlow\runs"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

foreach ($arraySize in @("8x8", "16x16", "32x32")) {
    $existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object {
            $_.CommandLine -like "*train_same_frequency.py*" -and
            $_.CommandLine -like ("*--array-size " + $arraySize + "*")
        }
    if ($existing) {
        throw "A same-frequency process for $arraySize already exists."
    }

    $manifest = Join-Path $manifestRoot ("manifest_samefreq_6.7ghz_{0}_0deg.jsonl" -f $arraySize)
    $runDir = Join-Path $runRoot ("srm_samefreq_6.7_to_6.7_{0}_0deg_lite" -f $arraySize)
    $stdout = Join-Path $runRoot ("samefreq_6.7_single_beam_{0}_stdout.log" -f $arraySize)
    $stderr = Join-Path $runRoot ("samefreq_6.7_single_beam_{0}_stderr.log" -f $arraySize)
    $arguments = @(
        "-u",
        (Join-Path $repoRoot "train_same_frequency.py"),
        "--dataset-root", $datasetRoot,
        "--manifest-path", $manifest,
        "--height-stats-path", $heightStats,
        "--run-root", $runDir,
        "--array-size", $arraySize,
        "--model-size", "lite",
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
    Write-Output ("started {0}: PID {1}" -f $arraySize, $process.Id)
}
