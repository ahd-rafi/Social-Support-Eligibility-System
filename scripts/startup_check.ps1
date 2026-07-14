# Startup Health Check Script
# Run this before starting the application

Write-Host "=" * 80
Write-Host "SYSTEM HEALTH CHECK"
Write-Host "=" * 80

$allGood = $true

# Check 1: Docker Desktop
Write-Host "`n[1/5] Checking Docker Desktop..."
try {
    docker ps > $null 2>&1
    Write-Host "  ✓ Docker Desktop is running"
} catch {
    Write-Host "  ✗ Docker Desktop is NOT running"
    Write-Host "    → Start Docker Desktop and wait 30 seconds"
    $allGood = $false
}

# Check 2: Docker Containers
Write-Host "`n[2/5] Checking Database Containers..."
$containers = @("social-support-postgres", "social-support-mongo", "social-support-qdrant", "social-support-neo4j")
foreach ($container in $containers) {
    $status = docker ps --filter "name=$container" --format "{{.Status}}" 2>$null
    if ($status -like "*Up*") {
        Write-Host "  ✓ $container is running"
    } else {
        Write-Host "  ✗ $container is NOT running"
        Write-Host "    → Run: docker start $container"
        $allGood = $false
    }
}

# Check 3: Ollama
Write-Host "`n[3/5] Checking Ollama..."
try {
    $ollamaStatus = ollama ps 2>&1
    if ($ollamaStatus -match "phi4-mini") {
        Write-Host "  ✓ Ollama is running with phi4-mini loaded"
        if ($ollamaStatus -match "100% GPU") {
            Write-Host "    ✓ Using GPU (optimal)"
        } elseif ($ollamaStatus -match "100% CPU") {
            Write-Host "    ⚠ Using CPU (slow - GPU issue pending)"
        }
    } else {
        Write-Host "  ⚠ Ollama running but no model loaded"
        Write-Host "    → Model will load on first request"
    }
} catch {
    Write-Host "  ✗ Ollama is NOT running"
    Write-Host "    → Start Ollama from Start Menu"
    $allGood = $false
}

# Check 4: Tesseract OCR
Write-Host "`n[4/5] Checking Tesseract OCR..."
try {
    $tesseractVersion = tesseract --version 2>&1 | Select-Object -First 1
    Write-Host "  ✓ Tesseract installed: $($tesseractVersion -replace 'tesseract ','')"
} catch {
    Write-Host "  ✗ Tesseract is NOT installed"
    Write-Host "    → Run: .\scripts\install_tesseract.ps1"
    $allGood = $false
}

# Check 5: Python Virtual Environment
Write-Host "`n[5/5] Checking Python Environment..."
if ($env:VIRTUAL_ENV) {
    Write-Host "  ✓ Virtual environment activated: $env:VIRTUAL_ENV"
} else {
    Write-Host "  ✗ Virtual environment NOT activated"
    Write-Host "    → Run: .\.venv\Scripts\Activate.ps1"
    $allGood = $false
}

# Summary
Write-Host "`n" + ("=" * 80)
if ($allGood) {
    Write-Host "✓ ALL CHECKS PASSED - System ready!"
    Write-Host ""
    Write-Host "Start the application:"
    Write-Host "  Terminal 1: uvicorn src.api.main:app --reload --port 8000"
    Write-Host "  Terminal 2: streamlit run frontend/app.py --server.port 8501"
} else {
    Write-Host "✗ SOME CHECKS FAILED - Fix issues above before starting"
}
Write-Host ("=" * 80)

pause
