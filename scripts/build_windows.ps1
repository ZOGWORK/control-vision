# Build the Windows one-folder release, run its smoke test and zip it.
# Requires a clean tracked tree so SOURCE.txt names the exact source commit.
param(
    # Seconds the frozen smoke test may take before it is stopped.
    [int]$SmokeTimeoutSeconds = 900
)
$ErrorActionPreference = "Stop"

# Windows PowerShell turns a native program's stderr into a terminating error
# under "Stop" whenever output is redirected (as in CI logs). PyInstaller logs
# to stderr, so run native tools with stderr merged as plain text and check
# their exit code instead.
function Invoke-Native([scriptblock]$Command) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $saved }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    $buildTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $python = if ($env:CONTROL_VISION_PYTHON) { $env:CONTROL_VISION_PYTHON }
              elseif (Test-Path -LiteralPath (Join-Path $repoRoot ".venv\Scripts\python.exe")) {
                  Join-Path $repoRoot ".venv\Scripts\python.exe"
              } else { "python" }
    $changes = @(git status --porcelain --untracked-files=normal)
    if ($LASTEXITCODE -ne 0 -or $changes.Count -ne 0) {
        throw "Commit tracked changes before packaging so SOURCE.txt identifies the exact source."
    }
    $commit = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[0-9a-f]{40}$') {
        throw "Could not determine the source commit."
    }
    $releaseDir = Join-Path $repoRoot "build\release"
    if (Test-Path -LiteralPath $releaseDir) { Remove-Item -LiteralPath $releaseDir -Recurse -Force }
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
    $wrapperVersion = & $python -c "import importlib.metadata; print(importlib.metadata.version('imageio-ffmpeg'))"
    if ($LASTEXITCODE -ne 0 -or $wrapperVersion -ne "0.6.0") {
        throw "Unexpected imageio-ffmpeg package version; update binary provenance before packaging."
    }
    & $ffmpeg -hide_banner -L | Set-Content -LiteralPath (Join-Path $releaseDir "FFMPEG_LICENSE.txt") -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Could not read bundled FFmpeg license." }
    & $ffmpeg -version | Set-Content -LiteralPath (Join-Path $releaseDir "FFMPEG_BUILD.txt") -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Could not identify bundled FFmpeg build." }
    $ffmpegVersion = (Get-Content -LiteralPath (Join-Path $releaseDir "FFMPEG_BUILD.txt") -TotalCount 1)
    if ($ffmpegVersion -notmatch '^ffmpeg version 7\.1-essentials_build-www\.gyan\.dev') {
        throw "Unexpected FFmpeg build; update the documented binary source and license before packaging."
    }
    $ffmpegHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ffmpeg).Hash.ToLowerInvariant()
    @"
Bundled FFmpeg binary provenance
Wrapper package: imageio-ffmpeg $wrapperVersion (requirements.txt)
Binary: $ffmpegVersion
SHA-256: $ffmpegHash
Provider release: https://github.com/GyanD/codexffmpeg/releases/tag/7.1
FFmpeg source commit identified by that release: https://github.com/FFmpeg/FFmpeg/tree/b08d7969c550a804a59511c7b83f2dd8cc0499b8
Build configuration: FFMPEG_BUILD.txt in this folder
License: FFMPEG_LICENSE.txt and licenses/GPL-3.0.txt in this folder

The release maintainer must separately ensure corresponding source for the
FFmpeg build and its linked libraries is provided with any public binary.
"@ | Set-Content -LiteralPath (Join-Path $releaseDir "FFMPEG_SOURCE.txt") -Encoding utf8

    Invoke-Native { & $python -m PyInstaller --noconfirm control_vision.spec }
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
    $exe = Join-Path $repoRoot "dist\ControlVision\ControlVision.exe"
    if (-not (Test-Path -LiteralPath $exe)) { throw "PyInstaller did not produce ControlVision.exe." }
    $packageDir = Split-Path -Parent $exe

    # The same notices are inside _internal; copy them beside the executable
    # so they are visible to anyone who opens the folder.
    Copy-Item -LiteralPath (Join-Path $repoRoot "LICENSE") -Destination $packageDir
    $licenseDir = Join-Path $packageDir "licenses"
    New-Item -ItemType Directory -Force -Path $licenseDir | Out-Null
    Copy-Item -Path (Join-Path $repoRoot "licenses\*.txt") -Destination $licenseDir
    Copy-Item -LiteralPath (Join-Path $releaseDir "licenses\python-packages") -Destination $licenseDir -Recurse
    Copy-Item -LiteralPath (Join-Path $repoRoot "docs\THIRD_PARTY_NOTICES.md") -Destination $packageDir
    foreach ($name in "SOURCE.txt", "FFMPEG_LICENSE.txt", "FFMPEG_BUILD.txt", "FFMPEG_SOURCE.txt") {
        Copy-Item -LiteralPath (Join-Path $releaseDir $name) -Destination $packageDir
    }
    $buildSeconds = [math]::Round($buildTimer.Elapsed.TotalSeconds, 1)

    # The executable is windowed, so wait on the process explicitly and read
    # the JSON report it writes instead of console output.
    $smokeReport = Join-Path $repoRoot "build\smoke-report.json"
    if (Test-Path -LiteralPath $smokeReport) { Remove-Item -LiteralPath $smokeReport -Force }
    $previousQtPlatform = $env:QT_QPA_PLATFORM
    $env:QT_QPA_PLATFORM = "offscreen"
    try {
        $smokeTimer = [System.Diagnostics.Stopwatch]::StartNew()
        $process = Start-Process -FilePath $exe -ArgumentList @("--smoke-test", "--report", "`"$smokeReport`"") `
            -PassThru -NoNewWindow
        $null = $process.Handle  # Keep the handle so ExitCode is available after exit.
        if (-not $process.WaitForExit($SmokeTimeoutSeconds * 1000)) {
            Stop-Process -Id $process.Id -Force
            throw "Frozen application smoke test did not finish within $SmokeTimeoutSeconds s."
        }
        $smokeSeconds = [math]::Round($smokeTimer.Elapsed.TotalSeconds, 1)
        if (Test-Path -LiteralPath $smokeReport) { Get-Content -LiteralPath $smokeReport | Write-Output }
        if ($process.ExitCode -ne 0) {
            throw "Frozen application smoke test failed with exit code $($process.ExitCode)."
        }
    } finally {
        if ($null -eq $previousQtPlatform) {
            Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
        } else {
            $env:QT_QPA_PLATFORM = $previousQtPlatform
        }
    }

    $zip = Join-Path $repoRoot "dist\ControlVision-windows-x64-$($commit.Substring(0, 12)).zip"
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    & (Join-Path $env:SystemRoot "System32\tar.exe") -a -c -f $zip -C (Join-Path $repoRoot "dist") ControlVision
    if ($LASTEXITCODE -ne 0) { throw "Could not create $zip." }
    $zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
    "$zipHash  $(Split-Path -Leaf $zip)" | Set-Content -LiteralPath "$zip.sha256" -Encoding ascii
    $files = @(Get-ChildItem -LiteralPath $packageDir -Recurse -File)
    $folderBytes = ($files | Measure-Object -Property Length -Sum).Sum
    Write-Output ("Verified build output: {0}" -f $exe)
    Write-Output ("Source commit: {0}" -f $commit)
    Write-Output ("Build time: {0} s; frozen smoke test: {1} s" -f $buildSeconds, $smokeSeconds)
    Write-Output ("Folder: {0} files, {1:N1} MiB" -f $files.Count, ($folderBytes / 1MB))
    Write-Output ("Zip: {0} ({1:N1} MiB)" -f $zip, ((Get-Item -LiteralPath $zip).Length / 1MB))
    Write-Output ("Zip SHA-256: {0}" -f $zipHash)
} finally {
    Pop-Location
}
