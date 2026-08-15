param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$BatchRoot,
    [Parameter(Mandatory = $true)][int]$CurrentJobPid,
    [Parameter(Mandatory = $true)][string]$PythonExe
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$root = (Resolve-Path -LiteralPath $BatchRoot).Path
$stopStatePath = Join-Path $root 'stop_after_current.json'
$followupStatePath = Join-Path $root 'two_seed_followup_state.json'

function Write-State([string]$Status, [string]$Detail) {
    [ordered]@{
        status = $Status
        detail = $Detail
        updated_at = (Get-Date).ToString('o')
        current_job_pid = $CurrentJobPid
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $followupStatePath
}

Write-State 'waiting_for_training' 'Waiting for CSJ seed 2027 to complete.'
if (Get-Process -Id $CurrentJobPid -ErrorAction SilentlyContinue) {
    Wait-Process -Id $CurrentJobPid
}

$stopCompleted = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if (Test-Path -LiteralPath $stopStatePath) {
        $state = Get-Content -Raw -Encoding UTF8 -LiteralPath $stopStatePath | ConvertFrom-Json
        if ($state.status -eq 'completed') {
            $stopCompleted = $true
            break
        }
    }
    Start-Sleep -Seconds 2
}
if (-not $stopCompleted) {
    Write-State 'failed' 'Training stopped, but the batch stop finalizer did not complete.'
    throw 'Batch stop finalizer did not complete within 60 seconds.'
}

Write-State 'running_diagnostics' 'Running 12 CS0/CSJ diagnostics for seeds 2026 and 2027.'
& $PythonExe (Join-Path $repo 'scripts\run_cstrong_diagnostics.py') `
    --root $root `
    --seeds 2026 2027 `
    --arms CS0 CSJ `
    --device cuda `
    --workers 0 `
    --max-samples 1000 `
    --batch-size 32
if ($LASTEXITCODE -ne 0) {
    Write-State 'failed' "Diagnostics exited with code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-State 'running_analysis' 'Diagnostics completed; generating tables, figures and Chinese report.'
& $PythonExe (Join-Path $repo 'scripts\analyze_cstrong_formal.py') `
    --input-root $root `
    --seeds 2026 2027
if ($LASTEXITCODE -ne 0) {
    Write-State 'failed' "Analysis exited with code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-State 'completed' 'Diagnostics and analysis completed successfully.'
