$ErrorActionPreference = "Stop"
python -m PyInstaller --clean --noconfirm control_vision.spec
Write-Output "Build output: dist/ControlVision/ControlVision.exe"
