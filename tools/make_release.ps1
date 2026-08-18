# NpuEmbeddings -- assemble a downloadable release.
# SPDX-License-Identifier: Apache-2.0
#
# A release is everything a user needs and nothing they do not: the executable
# and the compiled NPU design. About 1.2 MB.
#
#   npuembed.exe            ~0.6 MB
#   gemm_rtp/               ~0.6 MB  one xclbin + 16 instruction streams
#
# THE MODEL IS NOT IN HERE, ON PURPOSE
# ------------------------------------
# The weights are sentence-transformers/all-MiniLM-L6-v2. Redistributing them
# would be permitted -- that model is Apache-2.0 -- but it would be the worse
# deal for everyone: a 66 MB binary blob in a stranger's zip that nobody can
# practically check against the original, going stale the moment upstream
# changes, and hiding the model card the user ought to read.
#
# So get-model.cmd fetches the two files from HuggingFace, verifies the
# checkpoint's sha256 against the value this project pinned its goldens to,
# and calls `npuembed --prepare-model`. No Python: the container builder is in
# the executable (runtime/src/npue_pack.cpp), and it is verified byte-identical
# to the reference packer by tools/verify_pack_parity.py.
#
# The manifest records the sha256 of every file and the numbers this build was
# verified at, so a downloaded release can be checked against the run that
# produced it rather than trusted.
#
# Usage:
#   .\tools\make_release.ps1 -Version v0.1.0
#   .\tools\make_release.ps1 -Version v0.1.0 -Artifacts artifacts_b128il

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Artifacts = "artifacts_b128il",
    [string]$OutDir = "dist"
)

$ErrorActionPreference = "Stop"
$REPO = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $REPO "$OutDir\npuembeddings-$Version"

$exe = Join-Path $REPO "runtime\build\npuembed.exe"
$design = Join-Path $REPO "runtime\$Artifacts\gemm_rtp"
$pin = Join-Path $REPO "models\all-MiniLM-L6-v2\CHECKPOINT.json"

foreach ($p in @($exe, $design, $pin)) {
    if (-not (Test-Path $p)) {
        throw "missing $p -- see BUILD.md (the release cannot be assembled from a partial build)"
    }
}

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
New-Item -ItemType Directory -Force -Path "$stage\models" | Out-Null

Copy-Item $exe $stage
Copy-Item -Recurse $design "$stage\gemm_rtp"
Copy-Item (Join-Path $REPO "LICENSE") $stage
if (Test-Path (Join-Path $REPO "README.md")) {
    Copy-Item (Join-Path $REPO "README.md") $stage
}

# get-model.cmd: fetch the checkpoint from its source and build the container.
# curl ships with Windows 10 and later, so this needs nothing installed.
$pinned = (Get-Content $pin -Raw | ConvertFrom-Json).sha256
$repoId = (Get-Content $pin -Raw | ConvertFrom-Json).repo_id
@"
@echo off
rem NpuEmbeddings -- fetch all-MiniLM-L6-v2 and build the model container.
rem
rem The weights are not redistributed with this release. They come from
rem   https://huggingface.co/$repoId
rem which is Apache-2.0 licensed; please read its model card. This script
rem downloads them, checks the checksum, and builds models\all-MiniLM-L6-v2.npue.
setlocal
set BASE=https://huggingface.co/$repoId/resolve/main
set DIR=%~dp0models\src
if not exist "%DIR%" mkdir "%DIR%"

echo Downloading from %BASE% ...
curl -L -f --progress-bar -o "%DIR%\model.safetensors" "%BASE%/model.safetensors" || goto :fail
curl -L -f --progress-bar -o "%DIR%\vocab.txt"         "%BASE%/vocab.txt"         || goto :fail
curl -L -f --progress-bar -o "%DIR%\config.json"       "%BASE%/config.json"       || goto :fail

echo.
echo Verifying checkpoint ...
for /f "skip=1 tokens=1" %%H in ('certutil -hashfile "%DIR%\model.safetensors" SHA256') do (
  if not defined GOT set GOT=%%H
)
if /I not "%GOT%"=="$pinned" (
  echo   MISMATCH
  echo     expected $pinned
  echo     got      %GOT%
  echo   These are not the weights this build was verified against. Stopping.
  goto :fail
)
echo   ok  %GOT%

echo.
"%~dp0npuembed.exe" --prepare-model "%DIR%" "%~dp0models\all-MiniLM-L6-v2.npue" || goto :fail
echo.
echo Done. Run run-server.cmd
exit /b 0

:fail
echo.
echo FAILED -- see the messages above.
exit /b 1
"@ | Set-Content -Path "$stage\get-model.cmd" -Encoding ascii

# The layout the runtime expects when pointed at a release directory: it takes
# a root and looks for models/*.npue and <root>/runtime/<artifacts>/gemm_rtp,
# so a release ships a tiny launcher rather than asking the user to recreate
# that tree by hand.
@'
@echo off
rem NpuEmbeddings -- start the embeddings server on http://127.0.0.1:8420
rem Usage: run-server.cmd [port]
setlocal
set PORT=%1
if "%PORT%"=="" set PORT=8420
"%~dp0npuembed.exe" "%~dp0." --artifacts . --threads 24 --pipeline 2 --serve %PORT%
'@ | Set-Content -Path "$stage\run-server.cmd" -Encoding ascii

@'
@echo off
rem NpuEmbeddings -- embed one text file (one text per line) to out.f32
rem Usage: embed.cmd texts.txt out.f32
setlocal
"%~dp0npuembed.exe" "%~dp0." --artifacts . --threads 24 --pipeline 2 --embed %1 %2
'@ | Set-Content -Path "$stage\embed.cmd" -Encoding ascii

# Checksums over everything shipped.
$files = Get-ChildItem -Recurse -File $stage | Sort-Object FullName
$hashes = foreach ($f in $files) {
    $rel = $f.FullName.Substring($stage.Length + 1)
    [ordered]@{ file = $rel
                bytes = $f.Length
                sha256 = (Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower() }
}

$manifest = [ordered]@{
    name    = "NpuEmbeddings"
    version = $Version
    model   = "all-MiniLM-L6-v2"
    model_source = @{ repo = $repoId; sha256 = $pinned
                      note = "fetched by get-model.cmd; not redistributed" }
    hardware = "AMD Ryzen AI (XDNA2 / Strix Point) NPU"
    sequence_length = 64
    embedding_dim = 384
    artifacts_set = $Artifacts
    built_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    verified = [ordered]@{
        note = "figures from tasks/0033-0037; reproduce with tools/verify_*.py"
        worst_1_minus_cos_vs_huggingface = 1.086e-05
        tokenizer_exact_match = "6826/6826"
        mteb_delta_points = 0.04
        throughput_seq_per_s = 918
        energy_j_per_1000_seq = 44.0
    }
    files = $hashes
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content "$stage\manifest.json" -Encoding utf8

$zip = Join-Path $REPO "$OutDir\npuembeddings-$Version-win-x64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$stage\*" -DestinationPath $zip -CompressionLevel Optimal

$mb = (Get-Item $zip).Length / 1MB
Write-Host ""
Write-Host "  staged  $stage"
Write-Host ("  zip     {0}  ({1:N1} MB)" -f $zip, $mb)
Write-Host ("  files   {0}" -f $files.Count)
Write-Host ""
Write-Host "  Upload the zip as a GitHub release asset for tag $Version."
Write-Host "  Users need: AMD Ryzen AI driver + XRT, and the MSVC 2015-2022 redistributable."
Write-Host "  First run: get-model.cmd -- it downloads the weights and builds the container."
