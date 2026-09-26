# Final cycle: e010 experiment, then test submission. Each step is its own process writing
# directly to files (no pipes through this script), so nothing breaks if this script is stopped.
$wd = "C:\Users\chhav\OneDrive\Desktop\Buisness entity\business_entity_resolution"
Set-Location $wd
$env:PYTHONIOENCODING = "utf-8"
$py = "$wd\.venv\Scripts\python.exe"
$log = "$wd\e010_chain.log"
"[chain] experiment start $(Get-Date -Format HH:mm)" | Out-File $log -Encoding utf8
$p = Start-Process -FilePath $py -ArgumentList '-u', '-m', 'ber.run_experiment', 'configs/exp/e010_consensus.yaml' `
     -WorkingDirectory $wd -RedirectStandardOutput "$wd\e010.log" -RedirectStandardError "$wd\e010.err" -WindowStyle Hidden -PassThru -Wait
"[chain] experiment exit $($p.ExitCode) $(Get-Date -Format HH:mm)" | Out-File $log -Append -Encoding utf8
if ($p.ExitCode -ne 0) { exit 1 }
"[chain] submission start $(Get-Date -Format HH:mm)" | Out-File $log -Append -Encoding utf8
$p = Start-Process -FilePath $py -ArgumentList '-u', '-m', 'ber.make_submission', 'configs/exp/e010_consensus.yaml', '--note', 'e010-consensus-legal' `
     -WorkingDirectory $wd -RedirectStandardOutput "$wd\sub_e010.log" -RedirectStandardError "$wd\sub_e010.err" -WindowStyle Hidden -PassThru -Wait
"[chain] ALL DONE exit $($p.ExitCode) $(Get-Date -Format HH:mm)" | Out-File $log -Append -Encoding utf8
