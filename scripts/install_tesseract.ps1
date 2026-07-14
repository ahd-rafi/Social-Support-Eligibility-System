# Install Tesseract OCR on Windows
# This script downloads and installs Tesseract automatically

Write-Host "=" * 80
Write-Host "Installing Tesseract OCR for Windows"
Write-Host "=" * 80

$tesseractUrl = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.5.0.20250111.exe"
$installerPath = "$env:TEMP\tesseract-installer.exe"

Write-Host "`nStep 1: Downloading Tesseract installer..."
try {
    Invoke-WebRequest -Uri $tesseractUrl -OutFile $installerPath
    Write-Host "  ✓ Downloaded successfully"
} catch {
    Write-Host "  ✗ Download failed. Please download manually from:"
    Write-Host "    https://github.com/UB-Mannheim/tesseract/wiki"
    exit 1
}

Write-Host "`nStep 2: Installing Tesseract..."
Write-Host "  (Installation window will appear - click through the prompts)"
Write-Host "  IMPORTANT: Make sure 'Add to PATH' is checked!"
Write-Host ""

Start-Process -FilePath $installerPath -Wait

Write-Host "`nStep 3: Verifying installation..."
Start-Sleep -Seconds 2

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

try {
    $version = tesseract --version 2>&1 | Select-Object -First 1
    Write-Host "  ✓ Tesseract installed: $version"
} catch {
    Write-Host "  ✗ Tesseract not found in PATH"
    Write-Host ""
    Write-Host "Manual fix needed:"
    Write-Host "1. Add to PATH: C:\Program Files\Tesseract-OCR"
    Write-Host "2. Restart PowerShell"
    Write-Host "3. Test: tesseract --version"
}

Write-Host "`n" + ("=" * 80)
Write-Host "Next steps:"
Write-Host "1. RESTART YOUR POWERSHELL TERMINALS (both FastAPI and this one)"
Write-Host "2. Test: tesseract --version"
Write-Host "3. Restart FastAPI: uvicorn src.api.main:app --reload --port 8000"
Write-Host ("=" * 80)

pause
