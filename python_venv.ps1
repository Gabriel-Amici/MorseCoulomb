# Create virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r .\src\requirements.txt
pip install -e .\src

Write-Host "Virtual environment created and packages installed."
Write-Host "The virtual environment is now active in this PowerShell session."
