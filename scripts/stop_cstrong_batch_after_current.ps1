param(
    [Parameter(Mandatory = $true)][string]$BatchRoot,
    [Parameter(Mandatory = $true)][int]$BatchPid,
    [Parameter(Mandatory = $true)][int]$JobPid,
    [Parameter(Mandatory = $true)][string]$JobName
)

$ErrorActionPreference = 'Stop'
$resolvedRoot = (Resolve-Path -LiteralPath $BatchRoot).Path
$manifestPath = Join-Path $resolvedRoot 'batch_manifest.json'
$registryPath = Join-Path $resolvedRoot 'registry.csv'
$lockPath = Join-Path $resolvedRoot 'runner.lock'
$statePath = Join-Path $resolvedRoot 'stop_after_current.json'

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class CStrongProcessControl {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr OpenProcess(uint access, bool inheritHandle, int processId);
    [DllImport("ntdll.dll")]
    public static extern int NtSuspendProcess(IntPtr processHandle);
    [DllImport("kernel32.dll")]
    public static extern bool CloseHandle(IntPtr handle);
}
'@

$batch = Get-Process -Id $BatchPid -ErrorAction Stop
$job = Get-Process -Id $JobPid -ErrorAction Stop
$handle = [CStrongProcessControl]::OpenProcess(0x0800, $false, $BatchPid)
if ($handle -eq [IntPtr]::Zero) {
    throw "Could not open batch process $BatchPid for suspension."
}
try {
    $status = [CStrongProcessControl]::NtSuspendProcess($handle)
    if ($status -ne 0) {
        throw "NtSuspendProcess failed with status $status."
    }
}
finally {
    [CStrongProcessControl]::CloseHandle($handle) | Out-Null
}

[ordered]@{
    status = 'armed'
    batch_pid = $BatchPid
    job_pid = $JobPid
    job = $JobName
    armed_at = (Get-Date).ToString('o')
} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $statePath

Wait-Process -Id $JobPid
Start-Sleep -Seconds 2

$rows = Import-Csv -Encoding UTF8 -LiteralPath $registryPath
$jobRow = $rows | Where-Object { $_.job -eq $JobName } | Select-Object -First 1
if (-not $jobRow) {
    throw "Could not find $JobName in registry."
}
$jobManifestPath = Join-Path $jobRow.output 'manifest.json'
$jobManifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $jobManifestPath | ConvertFrom-Json
if ($jobManifest.status -ne 'completed') {
    throw "Current job exited without a completed manifest: status=$($jobManifest.status)"
}

Stop-Process -Id $BatchPid -Force -ErrorAction SilentlyContinue
$finishedAt = (Get-Date).ToString('o')
foreach ($row in $rows) {
    if ($row.job -eq $JobName) {
        $row.status = 'completed'
        $row.finished_at = $finishedAt
        $row.exit_code = '0'
        $row.note = 'completed; parent batch stopped by user before next job'
    }
    elseif ($row.status -eq 'pending') {
        $row.status = 'deferred_by_user'
        $row.note = 'deferred after CSJ seed 2027'
    }
}
$temporaryRegistry = "$registryPath.tmp"
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $temporaryRegistry
Move-Item -Force -LiteralPath $temporaryRegistry -Destination $registryPath

$batchManifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifestPath | ConvertFrom-Json
$batchManifest.status = 'stopped_by_user_after_current'
$batchManifest | Add-Member -Force NoteProperty stopped_at $finishedAt
$batchManifest | Add-Member -Force NoteProperty stopped_after_job $JobName
$batchManifest | Add-Member -Force NoteProperty completed_jobs (($rows | Where-Object status -eq 'completed').Count)
$batchManifest | Add-Member -Force NoteProperty deferred_jobs @($rows | Where-Object status -eq 'deferred_by_user' | ForEach-Object job)
$temporaryManifest = "$manifestPath.tmp"
$batchManifest | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 -LiteralPath $temporaryManifest
Move-Item -Force -LiteralPath $temporaryManifest -Destination $manifestPath

if (Test-Path -LiteralPath $lockPath) {
    $lock = Get-Content -Raw -Encoding UTF8 -LiteralPath $lockPath | ConvertFrom-Json
    if ([int]$lock.pid -eq $BatchPid) {
        Remove-Item -LiteralPath $lockPath
    }
}

[ordered]@{
    status = 'completed'
    batch_pid = $BatchPid
    job_pid = $JobPid
    job = $JobName
    finished_at = $finishedAt
} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $statePath
