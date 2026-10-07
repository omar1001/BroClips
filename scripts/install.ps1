<#
BroClips installer for Windows (SPEC section 10). Double-click install.bat - it runs this file.

  -Python <path>        use an existing Python 3.10+ instead of making .venv (packages are only checked there;
                        nothing is installed into it without asking)
  -ShortcutDir <dir>    where to put BroClips.lnk (default: your Desktop)
  -NoPip                do not install Python packages
  -NoAssets             do not download fonts, emoji or models
  -AssetsFrom <dir>     copy fonts/emoji/fribidi/models from this folder first (e.g. D:\OldBroClips\assets)
  -LocalSTT ask|yes|no  local speech-to-text (faster-whisper + NVIDIA libraries, about 1.5 GB)
  -Cutout ask|general|lite|no   the cut-out model for thumbnails (928 MB / 214 MB)
This file is plain ASCII on purpose (Windows PowerShell 5.1 reads it in the local code page).
#>
param(
    [string]$Python = "",
    [string]$ShortcutDir = "",
    [switch]$NoPip,
    [switch]$NoAssets,
    [string]$AssetsFrom = "",
    [ValidateSet("ask", "yes", "no")][string]$LocalSTT = "ask",
    [ValidateSet("ask", "general", "lite", "no")][string]$Cutout = "ask"
)
$ErrorActionPreference = "Continue"     # native programs write warnings to stderr; exit codes are checked instead
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Repo
$env:PYTHONUTF8 = "1"

function Say([string]$Text, [string]$Color = "White") { Write-Host $Text -ForegroundColor $Color }

function Ask-YesNo([string]$Question, [bool]$Default = $true) {
    $hint = if ($Default) { "[Y/n]" } else { "[y/N]" }
    $a = Read-Host "$Question $hint"
    if ([string]::IsNullOrWhiteSpace($a)) { return $Default }
    return $a.Trim().ToLower().StartsWith("y")
}

function Find-Python([string]$Exe, [string[]]$Extra) {
    # The full path of a working Python 3.10 or newer, else $null. (The Microsoft Store "python" stub fails here.)
    try {
        $out = & $Exe @Extra -c "import sys; print(sys.version_info[0], sys.version_info[1], sys.executable)" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
        $parts = ([string]($out | Select-Object -Last 1)).Split(" ", 3)
        if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10) { return $parts[2].Trim() }
    } catch { }
    return $null
}

function Has-Modules([string]$Py, [string]$Modules) {
    & $Py -c "import $Modules" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

Say ""
Say "=== BroClips installer ===" Cyan
Say "Folder: $Repo"

# ---- 1. Python 3.10+ ----
$py = $null
if ($Python) {
    $py = Find-Python $Python @()
    if (-not $py) {
        Say "The Python you gave ($Python) does not start, or it is older than 3.10." Red
        exit 1
    }
} else {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.10", "3.13")) {
            $py = Find-Python "py" @("-$v")
            if ($py) { break }
        }
    }
    if (-not $py -and (Get-Command python -ErrorAction SilentlyContinue)) { $py = Find-Python "python" @() }
}
if (-not $py) {
    Say "Python 3.10 or newer was not found." Red
    Say "Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run install.bat again."
    exit 1
}
Say "Python: $py" Green

# ---- 2. ffmpeg ----
function Has-FFmpeg { return [bool](Get-Command ffmpeg -ErrorAction SilentlyContinue) -and [bool](Get-Command ffprobe -ErrorAction SilentlyContinue) }
if (Has-FFmpeg) {
    Say "ffmpeg: found" Green
} else {
    Say "ffmpeg was not found. BroClips needs it to read and make videos." Yellow
    if ((Get-Command winget -ErrorAction SilentlyContinue) -and (Ask-YesNo "Install ffmpeg now with winget (Gyan.FFmpeg)?")) {
        winget install --id Gyan.FFmpeg -e
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    }
    if (Has-FFmpeg) {
        Say "ffmpeg: installed" Green
    } else {
        Say "ffmpeg is still missing. Install it later (https://www.gyan.dev/ffmpeg/builds/ - the 'full' build) and add" Yellow
        Say "its bin folder to PATH. If winget just installed it, close this window and run install.bat again." Yellow
    }
}

# ---- 3. The Python environment ----
$venvDir = Join-Path $Repo ".venv"
$record = Join-Path $venvDir "python-path.txt"     # tells run.bat which Python to use when -Python was given
if ($Python) {
    $envPy = $py
    New-Item -ItemType Directory -Force -Path $venvDir | Out-Null
    Set-Content -LiteralPath $record -Value $envPy -Encoding ASCII
    Say "Using your Python (no .venv): $envPy" Green
} else {
    $envPy = Join-Path $venvDir "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $envPy)) {
        Say "Making a private Python environment (.venv)..."
        & $py -m venv $venvDir
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $envPy)) { Say "Could not make .venv." Red; exit 1 }
    }
    if (Test-Path -LiteralPath $record) { Remove-Item -LiteralPath $record }
}

# ---- 4. Python packages ----
if ($NoPip) {
    Say "Skipping Python packages (-NoPip)." Yellow
} elseif ($Python -and (Has-Modules $envPy "numpy, PIL, psutil, onnxruntime")) {
    Say "Packages: already in your Python - nothing to install." Green
} else {
    if ($Python -and -not (Ask-YesNo "Some packages are missing in your Python. Install them into it?" $false)) {
        Say "Skipped. Install them yourself: python -m pip install -r requirements.txt" Yellow
    } else {
        Say "Installing the Python packages BroClips needs (a few minutes)..."
        & $envPy -m pip install --disable-pip-version-check -r (Join-Path $Repo "requirements.txt")
        if ($LASTEXITCODE -ne 0) { Say "Installing packages failed. Check the internet connection and run install.bat again." Red; exit 1 }
    }
}
if (-not $NoPip) {
    if (Has-Modules $envPy "faster_whisper") {
        Say "Local speech-to-text: already installed" Green
    } else {
        $wantLocal = $false
        if ($LocalSTT -eq "yes") { $wantLocal = $true }
        elseif ($LocalSTT -eq "ask") {
            Say ""
            Say "Local speech-to-text turns speech into text on THIS PC (free, private). Without it, use a key in Settings."
            $wantLocal = Ask-YesNo "Local speech-to-text (faster-whisper + NVIDIA libraries, ~1.5 GB)?"
        }
        if ($wantLocal) {
            & $envPy -m pip install --disable-pip-version-check -r (Join-Path $Repo "requirements-local.txt")
            if ($LASTEXITCODE -ne 0) { Say "Local speech-to-text could not be installed. You can still use a key, or try again later." Yellow }
        }
    }
}

# ---- 5. Fonts, emoji, Arabic letter fix, cut-out model ----
if ($NoAssets) {
    Say "Skipping downloads (-NoAssets)." Yellow
} else {
    Say ""
    Say "Downloading fonts, 3D emoji and the Arabic letter fix (about 5 MB, slowly on purpose)..."
    $assetArgs = @("-m", "broclips.assets", "--essential")
    if ($AssetsFrom) { $assetArgs += @("--from", $AssetsFrom) }
    & $envPy @assetArgs
    if ($LASTEXITCODE -ne 0) { Say "Some downloads failed. BroClips still works; press Download in Settings later." Yellow }
    $cut = $Cutout
    if ($cut -eq "ask") {
        Say ""
        Say "The cut-out model finds the person (or thing) in a picture, for thumbnails."
        $a = Read-Host "Download it now? [1] full, 928 MB (best)  [2] light, 214 MB  [N] not now"
        $cut = "no"
        if ($a -match "^\s*1") { $cut = "general" } elseif ($a -match "^\s*2") { $cut = "lite" }
    }
    if ($cut -ne "no") {
        $cutArgs = @("-m", "broclips.assets", "--cutout", $cut)
        if ($AssetsFrom) { $cutArgs += @("--from", $AssetsFrom) }
        & $envPy @cutArgs
        if ($LASTEXITCODE -ne 0) { Say "The cut-out model did not finish. Download it later in Settings (it resumes)." Yellow }
    }
}

# ---- 6. The Desktop shortcut ----
if (-not $ShortcutDir) { $ShortcutDir = [Environment]::GetFolderPath("Desktop") }
New-Item -ItemType Directory -Force -Path $ShortcutDir | Out-Null
$pyw = Join-Path (Split-Path -Parent $envPy) "pythonw.exe"
if (-not (Test-Path -LiteralPath $pyw)) {
    $pyw = $envPy
    Say "pythonw.exe was not found next to Python - the shortcut will also show a console window." Yellow
}
$lnk = Join-Path $ShortcutDir "BroClips.lnk"
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $pyw
$sc.Arguments = "-m broclips"
$sc.WorkingDirectory = $Repo
$sc.IconLocation = (Join-Path $Repo "broclips\web\broclips.ico") + ",0"
$sc.Description = "BroClips - your recordings to Shorts, a clean long video, titles and thumbnails"
$sc.Save()
Say "Shortcut: $lnk" Green

Say ""
Say "Done! Double-click 'BroClips' on your Desktop to start." Green
Say "The first time it opens Settings: choose your AI there (the Help page explains it in 2 minutes)."
exit 0
