# ResearchOps-Agent one-click start (Windows, no Docker).
# ASCII only: PowerShell 5.1 reads no-BOM scripts as ANSI, so this file must
# contain no non-ASCII characters. Double-click start_all.bat instead of
# running this file directly, or:  powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1
#
# Starts Qdrant (if found) + FastAPI backend + Next.js frontend in detached
# windows, frees ports 3000/8000/6333 first, clears the Next.js cache, and
# opens http://localhost:3000 after ~15s.

$repo = Split-Path -Parent $PSScriptRoot   # scripts\ -> repo root

Write-Host "== ResearchOps-Agent one-click start =="

# --- locate qdrant (PATH first, then the common D:\Users\<user>\tools spot) ---
$qdrant = (Get-Command qdrant -ErrorAction SilentlyContinue).Source
$qdDir = $null
if ($qdrant) {
  $qdDir = Split-Path -Parent $qdrant
} elseif (Test-Path "D:\Users\$env:USERNAME\tools\qdrant\qdrant.exe") {
  $qdrant = "D:\Users\$env:USERNAME\tools\qdrant\qdrant.exe"
  $qdDir = "D:\Users\$env:USERNAME\tools\qdrant"
}

# --- locate python (PATH first, then a common conda env) ---
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python -and (Test-Path "D:\Users\miniconda\envs\fastapi-env\python.exe")) {
  $python = "D:\Users\miniconda\envs\fastapi-env\python.exe"
}
if (-not $python) {
  Write-Host "python not found - install Python 3.12, run 'pip install -e .[dev]', and try again."
  exit 1
}

# --- locate node / npm (npm scripts spawn bare 'node' resolved via PATH) ---
$node = (Get-Command node -ErrorAction SilentlyContinue).Source
$npmCmd = (Get-Command npm -ErrorAction SilentlyContinue).Source
if (-not $node -and (Test-Path "D:\node.exe")) {
  $node = "D:\node.exe"
  if (-not $npmCmd) { $npmCmd = "D:\node_modules\npm\bin\npm-cli.js" }
}
if (-not $node -or -not $npmCmd) {
  Write-Host "node/npm not found - install Node.js 20+ (https://nodejs.org) and try again."
  exit 1
}
if (-not $env:Path.ToLower().Contains("d:\")) { $env:Path = "D:\;" + $env:Path }

# --- free ports ------------------------------------------------------------
Write-Host "[0/3] Freeing ports 3000/8000/6333 from any leftover process..."
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.LocalPort -in 3000, 8000, 6333 } |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

# --- start services --------------------------------------------------------
if ($qdrant) {
  Write-Host "[1/3] Starting Qdrant on port 6333..."
  Start-Process -FilePath $qdrant -ArgumentList "--disable-telemetry" -WorkingDirectory $qdDir
} else {
  Write-Host "[1/3] qdrant not found - skipping (RAG search will be unavailable)."
  Write-Host "       Install qdrant.exe on PATH, or 'docker run -p 6333:6333 qdrant/qdrant', then rerun."
}

Write-Host "[2/3] Starting FastAPI backend on port 8000..."
Start-Process -FilePath $python -ArgumentList "-m","uvicorn","researchops.server.main:app","--host","127.0.0.1","--port","8000" -WorkingDirectory $repo

Write-Host "[3/3] Clearing Next.js cache and starting frontend on port 3000..."
Remove-Item -Recurse -Force "$repo\web\.next" -ErrorAction SilentlyContinue
if (Test-Path "$repo\web\node_modules") {
  Start-Process -FilePath $node -ArgumentList $npmCmd,"run","dev" -WorkingDirectory "$repo\web"
} else {
  Write-Host "       node_modules missing - running npm install first (first time only)..."
  Start-Process -FilePath $node -ArgumentList $npmCmd,"install" -WorkingDirectory "$repo\web" -Wait
  Start-Process -FilePath $node -ArgumentList $npmCmd,"run","dev" -WorkingDirectory "$repo\web"
}

Write-Host ""
Write-Host "All windows are up. Opening http://localhost:3000 in 15s..."
Start-Sleep -Seconds 15
Start-Process "http://localhost:3000"
