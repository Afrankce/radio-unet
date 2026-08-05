param(
    [string]$RunSuffix = "_01x",
    [ValidateSet(0.1, 1.0)][double]$TrainScale = 0.1,
    [ValidateRange(1, 6)][int]$MaxConcurrent = 3
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$python = "D:\Anaconda3\envs\radioflow-win\python.exe"
$datasetRoot = "E:\datasets\MultiConfigRadiomap"
$manifestDir = "E:\datasets\MultiConfigRadiomap\manifests"
$runRoot = "E:\RadioFlow\runs\srm_6.7ghz_common8$RunSuffix"
$resultsRoot = "E:\RadioFlow\results\srm_6.7ghz_common8$RunSuffix"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$env:RADIOFLOW_RUN_ROOT = $runRoot
$env:MULTICONFIG_TRAIN_SCALE = "$TrainScale"
$arrays = @("8x8", "16x16", "32x32")
$modelSizes = @("lite", "large")
$device = "cuda:0"
$logDir = Join-Path $runRoot "_parallel_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Get-RuntimeJson {
    param([Parameter(Mandatory = $true)][string]$ArrayName,
          [Parameter(Mandatory = $true)][string]$ModelSize)
    $path = Join-Path $runRoot "$ArrayName\$ModelSize\training_runtime.json"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    return (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json)
}

function Get-RecordedEpochs {
    param([Parameter(Mandatory = $true)][string]$ArrayName,
          [Parameter(Mandatory = $true)][string]$ModelSize)
    $runtime = Get-RuntimeJson -ArrayName $ArrayName -ModelSize $ModelSize
    if ($null -eq $runtime) { return 0 }
    return [int]$runtime.completed_epochs
}

function Test-TrainingComplete {
    param([Parameter(Mandatory = $true)][string]$ArrayName,
          [Parameter(Mandatory = $true)][string]$ModelSize)
    $runtime = Get-RuntimeJson -ArrayName $ArrayName -ModelSize $ModelSize
    return ($null -ne $runtime -and $runtime.status -eq "complete")
}

function Start-TrainingProcess {
    param([Parameter(Mandatory = $true)][string]$ArrayName,
          [Parameter(Mandatory = $true)][string]$ModelSize,
          [switch]$Pilot)
    $controls = if ($Pilot) {
        @("--resume", "auto", "--stop-after-epoch", "5")
    } else {
        @("--resume", "auto")
    }
    $phase = if ($Pilot) { "pilot" } else { "full" }
    $stdout = Join-Path $logDir "${ArrayName}_${ModelSize}_${phase}.out.log"
    $stderr = Join-Path $logDir "${ArrayName}_${ModelSize}_${phase}.err.log"
    $arguments = @(
        "train_multiconfig.py",
        "--dataset-root", $datasetRoot,
        "--manifest-dir", $manifestDir,
        "--array", $ArrayName,
        "--model-size", $ModelSize,
        "--train-scale", "$TrainScale",
        "--run-root", $runRoot,
        "--device", $device
    ) + $controls
    return Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
}

function Start-EvaluationProcess {
    param([Parameter(Mandatory = $true)][string]$Command,
          [Parameter(Mandatory = $true)][string]$ArrayName,
          [Parameter(Mandatory = $true)][string]$ModelSize)
    $stdout = Join-Path $logDir "${ArrayName}_${ModelSize}_${Command}.out.log"
    $stderr = Join-Path $logDir "${ArrayName}_${ModelSize}_${Command}.err.log"
    $arguments = @(
        "evaluate_multiconfig.py", $Command,
        "--dataset-root", $datasetRoot,
        "--manifest-dir", $manifestDir,
        "--array", $ArrayName,
        "--model-size", $ModelSize,
        "--train-scale", "$TrainScale",
        "--run-root", $runRoot,
        "--results-root", $resultsRoot,
        "--device", $device
    )
    return Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
}

function Wait-Processes {
    param([Parameter(Mandatory = $true)][hashtable]$Running,
          [Parameter(Mandatory = $true)][string]$PhaseLabel)
    while ($Running.Count -gt 0) {
        Start-Sleep -Seconds 30
        $finished = @()
        foreach ($key in $Running.Keys) {
            $job = $Running[$key]
            if ($job.Process.HasExited) {
                if ($job.Process.ExitCode -ne 0) {
                    $log = Join-Path $logDir "${job.Array}_${job.Size}_${job.Phase}.err.log"
                    $tail = if (Test-Path -LiteralPath $log) { (Get-Content -LiteralPath $log -Tail 5) -join "`n" } else { "" }
                    throw "$PhaseLabel job $key failed with exit $($job.Process.ExitCode):`n$tail"
                }
                Write-Host "$(Get-Date -Format 'HH:mm:ss') $PhaseLabel done: $key"
                $finished += $key
            }
        }
        foreach ($key in $finished) { $Running.Remove($key) }
    }
}

$queue = [System.Collections.Queue]::new()
foreach ($array in $arrays) {
    foreach ($size in $modelSizes) {
        $queue.Enqueue(@($array, $size))
    }
}

# PHASE_TRAIN_PARALLEL: pilots first (to epoch 5), then full 200-epoch runs.
$running = @{}
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $MaxConcurrent) {
        $item = $queue.Dequeue()
        $key = "$($item[0])/$($item[1])"
        if (Test-TrainingComplete -ArrayName $item[0] -ModelSize $item[1]) {
            Write-Host "$(Get-Date -Format 'HH:mm:ss') already complete: $key"
            continue
        }
        $epochs = Get-RecordedEpochs -ArrayName $item[0] -ModelSize $item[1]
        $pilot = $epochs -lt 5
        $process = Start-TrainingProcess -ArrayName $item[0] -ModelSize $item[1] -Pilot:$pilot
        $running[$key] = @{
            Process = $process
            Array = $item[0]
            Size = $item[1]
            Phase = $(if ($pilot) { "pilot" } else { "full" })
        }
        Write-Host "$(Get-Date -Format 'HH:mm:ss') started $(if ($pilot) { 'pilot' } else { 'full' }) for $key"
    }
    if ($running.Count -eq 0) { break }
    Start-Sleep -Seconds 30
    $finished = @()
    foreach ($key in $running.Keys) {
        $job = $running[$key]
        if ($job.Process.HasExited) {
            if ($job.Process.ExitCode -ne 0) {
                $log = Join-Path $logDir "$($job.Array)_$($job.Size)_$($job.Phase).err.log"
                $tail = if (Test-Path -LiteralPath $log) { (Get-Content -LiteralPath $log -Tail 5) -join "`n" } else { "" }
                throw "training job $key failed with exit $($job.Process.ExitCode):`n$tail"
            }
            if ($job.Phase -eq "pilot" -and (Get-RecordedEpochs -ArrayName $job.Array -ModelSize $job.Size) -eq 5) {
                Write-Host "$(Get-Date -Format 'HH:mm:ss') pilot finished for $key, queuing full run"
                $queue.Enqueue(@($job.Array, $job.Size))
            }
            elseif (Test-TrainingComplete -ArrayName $job.Array -ModelSize $job.Size) {
                Write-Host "$(Get-Date -Format 'HH:mm:ss') full training complete: $key"
            }
            else {
                throw "training job $key ended in an unexpected state"
            }
            $finished += $key
        }
    }
    foreach ($key in $finished) { $running.Remove($key) }
}
Write-Host "All training complete."

# PHASE_CFG_SELECTION_PARALLEL
$running = @{}
foreach ($array in $arrays) {
    foreach ($size in $modelSizes) {
        $key = "$array/$size"
        if (-not (Test-TrainingComplete -ArrayName $array -ModelSize $size)) {
            throw "cannot select CFG for incomplete training $key"
        }
        $running[$key] = @{
            Process = Start-EvaluationProcess -Command "select-cfg" -ArrayName $array -ModelSize $size
            Array = $array
            Size = $size
            Phase = "select-cfg"
        }
    }
}
Wait-Processes -Running $running -PhaseLabel "CFG selection"

# PHASE_TEST_ONCE_PARALLEL
$running = @{}
foreach ($array in $arrays) {
    foreach ($size in $modelSizes) {
        $key = "$array/$size"
        $receipt = Join-Path $resultsRoot "$array\$size\run_manifest.json"
        if (Test-Path -LiteralPath $receipt -PathType Leaf) {
            Write-Host "test already complete: $key"
            continue
        }
        $running[$key] = @{
            Process = Start-EvaluationProcess -Command "test" -ArrayName $array -ModelSize $size
            Array = $array
            Size = $size
            Phase = "test"
        }
    }
}
Wait-Processes -Running $running -PhaseLabel "test evaluation"

# PHASE_SUMMARY
$summary = & $python @(
    "evaluate_multiconfig.py", "summarize",
    "--run-root", $runRoot,
    "--results-root", $resultsRoot
) 2>&1
$summaryExit = $LASTEXITCODE
$summary | ForEach-Object { Write-Host $_ }
if ($summaryExit -ne 0) {
    throw "benchmark summary failed with exit code $summaryExit"
}
Write-Host "Parallel benchmark pipeline complete."