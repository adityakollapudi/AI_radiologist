@echo off
title MEDVISION AI Shutdown
echo ========================================================
echo    Stopping MEDVISION AI Servers (Ports 8000 and 5173)
echo ========================================================
echo.

powershell -NoProfile -Command "
$ports = 8000, 5173
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conns) {
        $pids = $conns.OwningProcess | Select-Object -Unique
        foreach ($pid in $pids) {
            Write-Host ('Stopping process on port ' + $port + ' (PID: ' + $pid + ')...')
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
}
Write-Host 'Ports 8000 and 5173 released.'
"

echo.
echo Servers stopped successfully.
pause
