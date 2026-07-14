# Enable Ollama GPU Support
# Run this script as Administrator

Write-Host "Setting system environment variables for Ollama GPU support..."

# Set CUDA device visibility
[System.Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', '0', 'Machine')
Write-Host "  Set CUDA_VISIBLE_DEVICES=0"

# Force Ollama to use CUDA
[System.Environment]::SetEnvironmentVariable('OLLAMA_CUDA_VISIBLE_DEVICES', '0', 'Machine')
Write-Host "  Set OLLAMA_CUDA_VISIBLE_DEVICES=0"

# Optionally set number of GPU layers (all layers on GPU)
[System.Environment]::SetEnvironmentVariable('OLLAMA_NUM_GPU', '999', 'Machine')
Write-Host "  Set OLLAMA_NUM_GPU=999 (all layers)"

Write-Host "`n✓ Environment variables set!"
Write-Host "`nNext steps:"
Write-Host "1. Close this window"
Write-Host "2. Restart Ollama completely (Task Manager → End all ollama.exe → Start Ollama app)"
Write-Host "3. Test: ollama ps"
Write-Host "   Should show '100% GPU' instead of '100% CPU'"

pause
