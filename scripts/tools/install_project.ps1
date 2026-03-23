$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptDir "run_python.ps1"

& $runner "-m" "pip" "install" "-e" "R:\Frbot_Lift\source\Frbot_Lift"
