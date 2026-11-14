param(
    [ValidateSet('select','generate','evaluate','verify','report','all','doctor')][string]$Stage = 'all',
    [ValidateSet('lucid','ambilage')][string]$Method = 'ambilage',
    [string]$Config = 'src/experiments/config/pipeline.example.json',
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

# 优先使用 .venv 中的 Python
$runtimePython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) {
    $runtimePython = 'python'
}

Write-Host ">>> 执行 AmbiLAGE Pipeline: Stage=$Stage, Method=$Method, Device=$Device"
& $runtimePython -X utf8 -B -m src.experiments.pipeline $Stage --method $Method --config $Config --device $Device
exit $LASTEXITCODE
