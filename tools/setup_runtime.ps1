param(
    [string]$Python = 'python',
    [string]$IndexUrl = 'https://pypi.tuna.tsinghua.edu.cn/simple'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

Write-Host ">>> 准备 Python 虚拟运行环境 (.venv)..."
& $Python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw '创建虚拟运行环境失败' }

$pipExe = Join-Path $projectRoot '.venv\Scripts\python.exe'

if ($IndexUrl -ne '') {
    Write-Host ">>> 使用指定 PyPI 镜像源安装依赖: $IndexUrl"
    & $pipExe -m pip install -r requirements/requirements.txt --index-url $IndexUrl
} else {
    Write-Host ">>> 使用默认官方 PyPI 源安装依赖..."
    & $pipExe -m pip install -r requirements/requirements.txt
}

if ($LASTEXITCODE -ne 0) {
    throw '依赖安装失败，请检查网络设置'
}
Write-Host ">>> 运行环境依赖安装成功完成！"
