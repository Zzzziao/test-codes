param(
    [int]$Iterations = 2600,
    [int]$MaxSize = 0,
    [int]$Gpu = 0,
    [string]$Image = 'hazy input/1449.jpg',
    [string]$OutputDir = 'results/sid_gpu',
    [string]$PythonPath = 'D:\文稿\SID-UNN-gpu-env\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "CUDA Python environment not found: $PythonPath. Install a CUDA-enabled PyTorch build, or pass -PythonPath."
}

Push-Location $PSScriptRoot
try {
    & $PythonPath -u SID_gpu.py --image $Image --iterations $Iterations --max-size $MaxSize --gpu $Gpu --output-dir $OutputDir --no-plot
    if ($LASTEXITCODE -ne 0) { throw 'GPU SID-UNN experiment failed.' }
}
finally {
    Pop-Location
}
