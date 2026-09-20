param(
    [int]$Iterations = 2600,
    [int]$MaxSize = 0,
    [string]$Image = 'hazy input/1449.jpg',
    [string]$OutputDir = 'results/sid_joint'
)

$ErrorActionPreference = 'Stop'
$envRoot = 'D:\文稿\SID-UNN-env'
$env:TEMP = "$envRoot\tmp"
$env:TMP = "$envRoot\tmp"
$env:PIP_CACHE_DIR = "$envRoot\pip-cache"
$env:PYTHONPYCACHEPREFIX = "$envRoot\pycache"
$env:MPLCONFIGDIR = "$envRoot\matplotlib"
$env:TORCH_HOME = "$envRoot\torch-cache"
$env:MPLBACKEND = 'Agg'
$pythonPath = "$envRoot\Scripts\python.exe"

Push-Location $PSScriptRoot
try {
    & $pythonPath -u validate_wls.py
    if ($LASTEXITCODE -ne 0) { throw 'WLS validation failed; experiment was not started.' }
    & $pythonPath -u SID.py --image $Image --iterations $Iterations --max-size $MaxSize --output-dir $OutputDir --no-plot
    if ($LASTEXITCODE -ne 0) { throw 'SID-UNN experiment failed.' }
}
finally {
    Pop-Location
}
