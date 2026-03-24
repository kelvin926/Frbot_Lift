param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PythonArgs
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$projectRootAlias = "R:\Frbot_Lift"
$workspaceAlias = "R:"
$condaPython = "C:\Users\HYUNSEO\miniconda3\envs\env_isaaclab_2.3.1\python.exe"

if (-not (Test-Path $projectRootAlias)) {
    if (-not (Test-Path $workspaceAlias)) {
        $workspaceRoot = Split-Path -Parent $projectRoot
        cmd /c "subst $workspaceAlias `"$workspaceRoot`""
    }
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:CARB_APP_PATH = "E:\isaac-sim-5.1.0\kit"
$env:EXP_PATH = "E:\isaac-sim-5.1.0\apps"
$env:ISAAC_PATH = "E:\isaac-sim-5.1.0"
$env:FRBOT_LIFT_ROOT = $(if (Test-Path $projectRootAlias) { $projectRootAlias } else { $projectRoot })
$projectPythonRoot = Join-Path $env:FRBOT_LIFT_ROOT "source\Frbot_Lift"
$env:PYTHONPATH = "$projectPythonRoot;E:\isaac-sim-5.1.0\site;E:\isaac-sim-5.1.0\extscache\omni.usd.libs-1.0.1+69cbf6ad.wx64.r.cp311"
$env:PATH = "E:\isaac-sim-5.1.0\extscache\omni.usd.libs-1.0.1+69cbf6ad.wx64.r.cp311\bin;" + $env:PATH
$env:RESOURCE_NAME = "IsaacSim"
if (-not $env:FRBOT_LIFT_ASSET_PATH) {
    $env:FRBOT_LIFT_ASSET_PATH = Join-Path $env:FRBOT_LIFT_ROOT "assets\robots\lift_amr_fixedlift_runtime.usda"
} else {
    $env:FRBOT_LIFT_ASSET_PATH = $env:FRBOT_LIFT_ASSET_PATH.Trim()
}

if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
    throw "Pass a python script or module arguments to run_python.ps1."
}

& $condaPython @PythonArgs
