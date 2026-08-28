# Installs the NGX Telegram bot as a Windows service via NSSM. Run elevated.
$ErrorActionPreference = 'Stop'
$root   = '.'
$nssm   = Join-Path $root 'tools\nssm.exe'
$py     = Join-Path $root '.venv\Scripts\python.exe'
$log    = Join-Path $root 'data\bot_server.log'
$result = Join-Path $root 'tools\service_install_result.txt'
$svc    = 'NGXBot'

function Log($m) { Add-Content -Path $result -Value $m; Write-Host $m }
Set-Content -Path $result -Value "NGXBot service install @ $(Get-Date -Format o)"

# 1. Stop any manually-running bot instances so they don't double-poll Telegram
Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'bot_server' } |
  ForEach-Object { Log "stopping stray bot PID $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# 2. Remove existing service if re-running
if (Get-Service $svc -ErrorAction SilentlyContinue) {
  Log "removing existing $svc service"
  & $nssm stop $svc | Out-Null
  & $nssm remove $svc confirm | Out-Null
  Start-Sleep -Seconds 2
}

# 3. Install + configure
& $nssm install $svc $py '-m' 'notify.bot_server'
& $nssm set $svc AppDirectory $root
& $nssm set $svc AppEnvironmentExtra 'PYTHONIOENCODING=utf-8'
& $nssm set $svc DisplayName 'NGX Telegram Bot'
& $nssm set $svc Description 'Two-way Telegram bot for the Nigeria stocks decision system'
& $nssm set $svc Start SERVICE_AUTO_START
# log rotation: append + rotate at 10MB
& $nssm set $svc AppStdout $log
& $nssm set $svc AppStderr $log
& $nssm set $svc AppStdoutCreationDisposition 4
& $nssm set $svc AppStderrCreationDisposition 4
& $nssm set $svc AppRotateFiles 1
& $nssm set $svc AppRotateOnline 1
& $nssm set $svc AppRotateBytes 10485760
# restart on crash, throttle so a crash-loop backs off
& $nssm set $svc AppExit Default Restart
& $nssm set $svc AppRestartDelay 5000
& $nssm set $svc AppThrottle 5000

# 4. Start
& $nssm start $svc
Start-Sleep -Seconds 6
$s = Get-Service $svc -ErrorAction SilentlyContinue
Log "service status: $($s.Status)"
Log "startup type: $((Get-CimInstance Win32_Service -Filter "Name='$svc'").StartMode)"
