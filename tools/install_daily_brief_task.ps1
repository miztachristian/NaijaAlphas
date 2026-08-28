# Registers the NGX daily-brief scheduled task (MON-FRI 09:30, SYSTEM). Run elevated.
$ErrorActionPreference = 'Stop'
$root = '.'
$bat  = Join-Path $root '_run_daily.bat'
$res  = Join-Path $root 'tools\daily_task_install_result.txt'
$name = 'NGX Decision System'

Set-Content -Path $res -Value "daily-brief task install @ $(Get-Date -Format o)"

if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
  Unregister-ScheduledTask -TaskName $name -Confirm:$false
  Add-Content $res "removed existing task"
}

$action  = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$bat`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 09:30
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
              -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
              -MultipleInstances IgnoreNew -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
  -Principal $principal -Settings $settings `
  -Description 'Runs daily_ingest.py (full pipeline + Telegram daily brief) MON-FRI 09:30' | Out-Null

$t = Get-ScheduledTask -TaskName $name
$info = Get-ScheduledTaskInfo -TaskName $name
Add-Content $res "registered: $($t.TaskName)  state=$($t.State)"
Add-Content $res "next run: $($info.NextRunTime)"
Add-Content $res "principal: $($t.Principal.UserId)  runlevel=$($t.Principal.RunLevel)"
