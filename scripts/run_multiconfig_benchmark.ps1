param(
    [string]$RunSuffix = "",
    [ValidateSet(0.1, 1.0)][double]$TrainScale = 1.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:RADIOFLOW_RUN_ROOT = $runRoot
$env:MULTICONFIG_TRAIN_SCALE = "$TrainScale"

$python = "D:\Anaconda3\envs\radioflow-win\python.exe"
$datasetRoot = "E:\datasets\MultiConfigRadiomap"
$manifestDir = "E:\datasets\MultiConfigRadiomap\manifests"
$runRoot = "E:\RadioFlow\runs\srm_6.7ghz_common8$RunSuffix"
$resultsRoot = "E:\RadioFlow\results\srm_6.7ghz_common8$RunSuffix"
$arrays = @("8x8", "16x16", "32x32")
$modelSizes = @("lite", "large")
$device = "cuda:0"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$largeGate = Join-Path $runRoot "_hardware\large_hardware_gate.json"

function Invoke-JsonCommand {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $outputLines = @(& $python @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    foreach ($line in $outputLines) {
        Write-Host ([string]$line)
    }
    if ($exitCode -ne 0) {
        throw "Python command failed with exit code $exitCode`: $($Arguments -join ' ')"
    }
    for ($index = $outputLines.Count - 1; $index -ge 0; $index--) {
        try {
            return ([string]$outputLines[$index] | ConvertFrom-Json -ErrorAction Stop)
        }
        catch {
            continue
        }
    }
    throw "Python command produced no JSON result: $($Arguments -join ' ')"
}

function Invoke-CheckedCommand {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & $python @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Python command failed with exit code $exitCode`: $($Arguments -join ' ')"
    }
}

function Invoke-Training {
    param(
        [Parameter(Mandatory = $true)][string]$ArrayName,
        [Parameter(Mandatory = $true)][string]$ModelSize,
        [Parameter(Mandatory = $true)][string[]]$Controls
    )

    $arguments = @(
        "train_multiconfig.py",
        "--dataset-root", $datasetRoot,
        "--manifest-dir", $manifestDir,
        "--array", $ArrayName,
        "--model-size", $ModelSize,
        "--train-scale", "$TrainScale",
        "--run-root", $runRoot,
        "--device", $device
    ) + $Controls
    return Invoke-JsonCommand -Arguments $arguments
}

function Invoke-Evaluation {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string]$ArrayName,
        [Parameter(Mandatory = $true)][string]$ModelSize
    )

    return Invoke-JsonCommand -Arguments @(
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
}

function Get-RecordedEpoch {
    param([Parameter(Mandatory = $true)][string]$ArrayName,
          [Parameter(Mandatory = $true)][string]$ModelSize)

    $runtimePath = Join-Path $runRoot "$ArrayName\$ModelSize\training_runtime.json"
    if (-not (Test-Path -LiteralPath $runtimePath -PathType Leaf)) {
        return 0
    }
    $runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
    return [int]$runtime.completed_epochs
}

function Test-TrainingComplete {
    param([Parameter(Mandatory = $true)][string]$ArrayName,
          [Parameter(Mandatory = $true)][string]$ModelSize)

    $runtimePath = Join-Path $runRoot "$ArrayName\$ModelSize\training_runtime.json"
    if (-not (Test-Path -LiteralPath $runtimePath -PathType Leaf)) {
        return $false
    }
    $runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
    return $runtime.status -eq "complete"
}

Push-Location $repoRoot
try {
    $largeBlocked = Test-Path -LiteralPath $largeGate -PathType Leaf

    # PHASE_LITE_SMOKE
    $liteSmoke = Invoke-Training -ArrayName "8x8" -ModelSize "lite" -Controls @(
        "--resume", "none", "--smoke-optimizer-steps", "1"
    )
    if ($liteSmoke.status -ne "smoke_complete") {
        throw "Lite smoke did not complete"
    }

    # PHASE_LARGE_SMOKE
    if (-not $largeBlocked) {
        $largeSmoke = Invoke-Training -ArrayName "8x8" -ModelSize "large" -Controls @(
            "--resume", "none", "--smoke-optimizer-steps", "1"
        )
        if ($largeSmoke.status -eq "hardware_blocked") {
            $largeBlocked = $true
        }
        elseif ($largeSmoke.status -ne "smoke_complete") {
            throw "Large smoke returned an unexpected status"
        }
    }

    # PHASE_GPU_EVIDENCE
    Invoke-CheckedCommand -Arguments @(
        "-m", "pytest", "tests/test_multiconfig_gpu_smoke.py", "-m", "gpu", "-q"
    )

    # PHASE_LITE_PILOTS
    foreach ($arrayName in $arrays) {
        if ((Get-RecordedEpoch -ArrayName $arrayName -ModelSize "lite") -lt 5) {
            $pilot = Invoke-Training -ArrayName $arrayName -ModelSize "lite" -Controls @(
                "--resume", "auto", "--stop-after-epoch", "5"
            )
            if ([int]$pilot.completed_epochs -ne 5) {
                throw "Lite pilot failed to reach epoch 5 for $arrayName"
            }
        }
    }

    # PHASE_LARGE_PILOT
    if (-not $largeBlocked) {
        if ((Get-RecordedEpoch -ArrayName "8x8" -ModelSize "large") -lt 5) {
            $largePilot = Invoke-Training -ArrayName "8x8" -ModelSize "large" -Controls @(
                "--resume", "auto", "--stop-after-epoch", "5"
            )
            if ([int]$largePilot.completed_epochs -ne 5) {
                throw "Large pilot failed to reach epoch 5"
            }
        }
    }

    # PHASE_LITE_FULL
    foreach ($arrayName in $arrays) {
        if (-not (Test-TrainingComplete -ArrayName $arrayName -ModelSize "lite")) {
            $full = Invoke-Training -ArrayName $arrayName -ModelSize "lite" -Controls @(
                "--resume", "auto"
            )
            if ($full.status -ne "complete") {
                throw "Lite full training did not complete for $arrayName"
            }
        }
    }

    # PHASE_LARGE_FULL
    if (-not $largeBlocked) {
        foreach ($arrayName in $arrays) {
            if (-not (Test-TrainingComplete -ArrayName $arrayName -ModelSize "large")) {
                $full = Invoke-Training -ArrayName $arrayName -ModelSize "large" -Controls @(
                    "--resume", "auto"
                )
                if ($full.status -ne "complete") {
                    throw "Large full training did not complete for $arrayName"
                }
            }
        }
    }

    # PHASE_CFG_SELECTION
    foreach ($arrayName in $arrays) {
        foreach ($modelSize in $modelSizes) {
            if ($modelSize -eq "large" -and $largeBlocked) {
                continue
            }
            Invoke-Evaluation -Command "select-cfg" -ArrayName $arrayName -ModelSize $modelSize | Out-Null
        }
    }

    # PHASE_TEST_ONCE
    foreach ($arrayName in $arrays) {
        foreach ($modelSize in $modelSizes) {
            if ($modelSize -eq "large" -and $largeBlocked) {
                continue
            }
            $receipt = Join-Path $resultsRoot "$arrayName\$modelSize\run_manifest.json"
            if (-not (Test-Path -LiteralPath $receipt -PathType Leaf)) {
                Invoke-Evaluation -Command "test" -ArrayName $arrayName -ModelSize $modelSize | Out-Null
            }
        }
    }

    # PHASE_SUMMARY
    Invoke-JsonCommand -Arguments @(
        "evaluate_multiconfig.py", "summarize",
        "--run-root", $runRoot,
        "--results-root", $resultsRoot
    ) | Out-Null
}
finally {
    Pop-Location
}
