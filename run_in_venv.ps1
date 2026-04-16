# PowerShell script to activate the venv and run main.py
# Usage: .\run_in_venv.ps1 [args]

$venvPath = Join-Path $PSScriptRoot ".venv"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (-Not (Test-Path $activateScript)) {
    Write-Error "Virtual environment not found at $activateScript."
    exit 1
}

# Activate the venv in this session
. $activateScript

# Run main.py with any arguments passed to this script
python main.py @Args