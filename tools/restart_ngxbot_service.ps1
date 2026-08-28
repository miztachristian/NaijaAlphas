# Restarts the NGXBot service and reports status. Run elevated.
$ErrorActionPreference = 'Continue'
$root = '.'
$nssm = Join-Path $root 'tools\nssm.exe'
$log  = Join-Path $root 'data\bot_server.log'
$res  = Join-Path $root 'tools\service_restart_result.txt'

Set-Content -Path $res -Value "NGXBot restart @ $(Get-Date -Format o)"
# fresh log so we can confirm a clean start
Set-Content -Path $log -Value ''
& $nssm stop NGXBot | Out-Null
Start-Sleep -Seconds 2
& $nssm start NGXBot | Out-Null
Start-Sleep -Seconds 8
$s = Get-Service NGXBot
Add-Content $res "service status: $($s.Status)"
Add-Content $res "--- log tail ---"
if (Test-Path $log) { Get-Content $log -Tail 12 | ForEach-Object { Add-Content $res $_ } }
