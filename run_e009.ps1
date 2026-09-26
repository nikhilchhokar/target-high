# Waits for the running e008 submission (PID given as arg), then runs e009 experiment + submission.
param([int]$WaitPid)
$wd = "C:\Users\chhav\OneDrive\Desktop\Buisness entity\business_entity_resolution"
Set-Location $wd
$env:PYTHONIOENCODING = "utf-8"
$log = "$wd\e009_chain.log"
"[chain] waiting for PID $WaitPid $(Get-Date -Format HH:mm)" | Out-File $log -Encoding utf8
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
"[chain] loco start $(Get-Date -Format HH:mm)" | Out-File $log -Append -Encoding utf8
& "$wd\.venv\Scripts\python.exe" -u -m ber.loco configs/exp/e008_india.yaml *>> "$wd\loco.log"
"[chain] experiment start $(Get-Date -Format HH:mm)" | Out-File $log -Append -Encoding utf8
& "$wd\.venv\Scripts\python.exe" -u -m ber.run_experiment configs/exp/e009_legal.yaml *>> "$wd\e009.log"
"[chain] experiment exit $LASTEXITCODE $(Get-Date -Format HH:mm)" | Out-File $log -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { exit 1 }
"[chain] submission start $(Get-Date -Format HH:mm)" | Out-File $log -Append -Encoding utf8
& "$wd\.venv\Scripts\python.exe" -u -m ber.make_submission configs/exp/e009_legal.yaml --note "e009-legal-canon" *>> "$wd\sub_e009.log"
"[chain] ALL DONE exit $LASTEXITCODE $(Get-Date -Format HH:mm)" | Out-File $log -Append -Encoding utf8
