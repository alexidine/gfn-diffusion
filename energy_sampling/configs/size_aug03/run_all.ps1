# size_aug03 battery -- run sequentially from energy_sampling/.
# The (lo, 0.05) cell is ctrl_aug03/fx_static and is NOT re-run.
# Order: the sizing cell first (it is the load-bearing question), then the
# fwd_frac cell, then the both-cell.

$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe'
$ROOT = 'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling'
$LOGS = Join-Path $ROOT 'configs\size_aug03\run_logs'
$env:PYTHONPATH = 'C:\Users\mikem\Projects\mxt_gfn\mxtaltools;C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion'

$PER_ARM_TIMEOUT = 5400
$ARMS = @('sz_hi_f05', 'sz_lo_f20', 'sz_hi_f20')

if (-not (Test-Path $LOGS)) { New-Item -ItemType Directory -Force -Path $LOGS | Out-Null }
Set-Location $ROOT
$status = Join-Path $LOGS 'STATUS.txt'
"battery size_aug03 started $(Get-Date -Format s)" | Out-File -FilePath $status -Encoding utf8

foreach ($arm in $ARMS) {
    $out = Join-Path $LOGS "$arm.log"
    $err = Join-Path $LOGS "$arm.err.log"
    "START $arm $(Get-Date -Format s)" | Out-File -Append -FilePath $status -Encoding utf8
    $t0 = Get-Date
    $p = Start-Process -FilePath $PY -ArgumentList @('train.py', '--config', "configs\size_aug03\$arm.yaml") `
                       -WorkingDirectory $ROOT -PassThru -NoNewWindow `
                       -RedirectStandardOutput $out -RedirectStandardError $err
    $done = $p.WaitForExit($PER_ARM_TIMEOUT * 1000)
    if (-not $done) {
        "TIMEOUT $arm -- killing" | Out-File -Append -FilePath $status -Encoding utf8
        try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch {}
        Start-Sleep -Seconds 10
    }
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    # NB $p.ExitCode is often empty with -PassThru; judge arms by
    # 'Finished Training!' in the .log plus a _final.pt checkpoint.
    "END   $arm $(Get-Date -Format s)  minutes=$mins" | Out-File -Append -FilePath $status -Encoding utf8
    Start-Sleep -Seconds 20
}
"battery size_aug03 finished $(Get-Date -Format s)" | Out-File -Append -FilePath $status -Encoding utf8
