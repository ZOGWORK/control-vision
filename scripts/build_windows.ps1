$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    $python = if ($env:CONTROL_VISION_PYTHON) { $env:CONTROL_VISION_PYTHON }
              elseif (Test-Path -LiteralPath (Join-Path $repoRoot ".venv\Scripts\python.exe")) {
                  Join-Path $repoRoot ".venv\Scripts\python.exe"
              } else { "python" }
    $changes = @(git status --porcelain --untracked-files=no)
    if ($LASTEXITCODE -ne 0 -or $changes.Count -ne 0) {
        throw "Commit tracked changes before packaging so SOURCE.txt identifies the exact source."
    }
    $commit = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[0-9a-f]{40}$') {
        throw "Could not determine the source commit."
    }
    $releaseDir = Join-Path $repoRoot "build\release"
    New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
    @"
Control Vision corresponding source
Commit: $commit
Repository: https://github.com/ZOGWORK/control-vision
Source at this commit: https://github.com/ZOGWORK/control-vision/tree/$commit
License: GNU Affero General Public License v3.0 (LICENSE in this folder)

This build contains no trained BIOBUZZ robot model weights. User-supplied
weights and match videos are external to the application bundle.
"@ | Set-Content -LiteralPath (Join-Path $releaseDir "SOURCE.txt") -Encoding utf8

    $ffmpeg = & $python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $ffmpeg)) {
        throw "The pinned imageio-ffmpeg executable was not found."
    }
    & $ffmpeg -L | Set-Content -LiteralPath (Join-Path $releaseDir "FFMPEG_LICENSE.txt") -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Could not read bundled FFmpeg license." }
    & $ffmpeg -version | Set-Content -LiteralPath (Join-Path $releaseDir "FFMPEG_BUILD.txt") -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Could not identify bundled FFmpeg build." }

    & $python -m PyInstaller --clean --noconfirm control_vision.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
    $exe = Join-Path $repoRoot "dist\ControlVision\ControlVision.exe"
    if (-not (Test-Path -LiteralPath $exe)) { throw "PyInstaller did not produce ControlVision.exe." }
    $previousQtPlatform = $env:QT_QPA_PLATFORM
    $env:QT_QPA_PLATFORM = "offscreen"
    try {
        & $exe --smoke-test
        if ($LASTEXITCODE -ne 0) { throw "Frozen application smoke test failed with exit code $LASTEXITCODE." }
    } finally {
        if ($null -eq $previousQtPlatform) {
            Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
        } else {
            $env:QT_QPA_PLATFORM = $previousQtPlatform
        }
    }
    Write-Output "Verified build output: $exe"
} finally {
    Pop-Location
}
