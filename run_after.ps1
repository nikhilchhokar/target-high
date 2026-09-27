# Waits for a PID to exit, then runs one python module command (output to files). Launched via WMI.
param([int]$WaitPid, [string]$Args1, [string]$Out)
$wd = "C:\Users\chhav\OneDrive\Desktop\Buisness entity\business_entity_resolution"
Set-Location $wd
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
$env:PYTHONIOENCODING = "utf-8"
Start-Process -FilePath "$wd\.venv\Scripts\python.exe" -ArgumentList ($Args1 -split ' ') -WorkingDirectory $wd `
  -RedirectStandardOutput "$wd\$Out.log" -RedirectStandardError "$wd\$Out.err" -WindowStyle Hidden -Wait
