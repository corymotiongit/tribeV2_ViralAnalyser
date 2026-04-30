param(
    [switch]$NoBrowser
)

$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCandidates = @(
    (Join-Path $appDir ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent $appDir) "tribe311_clean\Scripts\python.exe"),
    "python"
)
$python = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -eq "python") {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) {
            $python = "python"
            break
        }
    } elseif (Test-Path $candidate) {
        $python = $candidate
        break
    }
}
$hostAddress = "127.0.0.1"
$preferredPort = 8000
$fallbackPorts = 8001..8010

function Test-AppUrl {
    param(
        [int]$Port
    )

    try {
        $response = Invoke-WebRequest -Uri ("http://{0}:{1}" -f $hostAddress, $Port) -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Test-TribeAppUrl {
    param(
        [int]$Port
    )

    try {
        $response = Invoke-WebRequest -Uri ("http://{0}:{1}" -f $hostAddress, $Port) -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 500) {
            return $false
        }
        $content = [string]$response.Content
        return $content -like "*TRIBE Review MVP*" -or $content -like "*Predict virality with Meta TRIBE v2*"
    } catch {
        return $false
    }
}

function Test-PortFree {
    param(
        [int]$Port
    )

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse($hostAddress), $Port)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Get-LaunchPort {
    if (Test-TribeAppUrl -Port $preferredPort) {
        return $preferredPort
    }

    if (Test-PortFree -Port $preferredPort) {
        return $preferredPort
    }

    foreach ($port in $fallbackPorts) {
        if (Test-PortFree -Port $port) {
            return $port
        }
    }

    return $null
}

if (-not $python) {
    Write-Host ""
    Write-Host "Python not found. Create .venv or install Python 3.11." -ForegroundColor Red
    exit 1
}

Set-Location $appDir

$port = Get-LaunchPort
if (-not $port) {
    Write-Host ""
    Write-Host "Could not find a free port from 8000 to 8010." -ForegroundColor Red
    exit 1
}

$url = "http://{0}:{1}" -f $hostAddress, $port

if (Test-TribeAppUrl -Port $port) {
    Write-Host ""
    Write-Host ("TRIBE Review already running on {0}" -f $url) -ForegroundColor Green
    if (-not $NoBrowser) {
        Start-Process $url
    }
    exit 0
}

Write-Host ""
Write-Host ("Starting TRIBE Review on {0}" -f $url) -ForegroundColor Cyan

if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($LaunchUrl)
        $deadline = (Get-Date).AddMinutes(3)
        while ((Get-Date) -lt $deadline) {
            try {
                $response = Invoke-WebRequest -Uri $LaunchUrl -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                    Start-Process $LaunchUrl
                    return
                }
            } catch {
            }
            Start-Sleep -Milliseconds 750
        }
    } -ArgumentList $url | Out-Null
}

& $python -m uvicorn app:app --host $hostAddress --port $port
