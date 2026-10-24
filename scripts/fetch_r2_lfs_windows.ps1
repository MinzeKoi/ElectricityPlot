param(
    [string]$CheckReport = "$PSScriptRoot\..\outputs\r2-start-20261023\lfs-check-retry2.json",
    [int]$Limit = 0,
    [int]$Throttle = 3,
    [int]$ConnectionTimeout = 20,
    [int]$OperationTimeout = 120,
    [ValidateSet('Lfs','Media')][string]$Source = 'Lfs',
    [string]$ResultReport = "$PSScriptRoot\..\outputs\r2-start-20261023\windows-lfs-result.json"
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$r2Workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$r2Check = Get-Content -LiteralPath $CheckReport -Raw | ConvertFrom-Json
if ($r2Check.ref -ne 'f266b762b752b32a31810d916d198ccc06f7ec51') {
    throw 'Use the pinned R1 restore check.'
}
foreach ($r2Item in $r2Check.missing) {
    if (-not $r2Item.path.StartsWith('outputs/accuracy/r1-fixed-20261022-v2/') -or
        $r2Item.sha256 -notmatch '^[a-f0-9]{64}$' -or $r2Item.bytes -le 0) {
        throw 'Unexpected R1 object scope.'
    }
}
$r2Missing = @($r2Check.missing | Group-Object sha256 | ForEach-Object { $_.Group[0] } | Sort-Object bytes)
if ($Limit -gt 0) { $r2Missing = @($r2Missing | Select-Object -First $Limit) }
if (-not $r2Missing.Count) { Write-Output 'No missing objects.'; exit 0 }
$r2Payload = @{
    operation = 'download'
    transfers = @('basic')
    ref = @{name = $r2Check.ref}
    objects = @($r2Missing | ForEach-Object { @{oid = $_.sha256; size = $_.bytes} })
} | ConvertTo-Json -Depth 6 -Compress
if ($Source -eq 'Media') {
    $r2Batch = @{objects = @($r2Missing | ForEach-Object {
        $r2EncodedPath = ($_.path.Split('/') | ForEach-Object { [uri]::EscapeDataString($_) }) -join '/'
        @{oid=$_.sha256;size=$_.bytes;actions=@{download=@{href=(
            'https://media.githubusercontent.com/media/MinzeKoi/ElectricityPlot/' + $r2Check.ref + '/' + $r2EncodedPath)}}}
    })}
} else {
    $r2Batch = Invoke-RestMethod -Uri 'https://github.com/MinzeKoi/ElectricityPlot.git/info/lfs/objects/batch' `
        -Method Post -ContentType 'application/vnd.git-lfs+json' -Headers @{Accept='application/vnd.git-lfs+json'} `
        -Body $r2Payload -ConnectionTimeoutSeconds $ConnectionTimeout -OperationTimeoutSeconds 60
}
$r2Expected = @{}
foreach ($r2Item in $r2Missing) { $r2Expected[$r2Item.sha256] = [long]$r2Item.bytes }
if (@($r2Batch.objects).Count -ne $r2Missing.Count) { throw 'Incomplete batch response.' }
$r2Seen = @{}
foreach ($r2Object in $r2Batch.objects) {
    if (-not $r2Expected.ContainsKey($r2Object.oid) -or $r2Seen.ContainsKey($r2Object.oid) -or
        [long]$r2Object.size -ne $r2Expected[$r2Object.oid] -or -not $r2Object.actions.download) {
        throw 'Unexpected or unavailable download object.'
    }
    $r2Seen[$r2Object.oid] = $true
}
$r2Cache = [IO.Path]::GetFullPath((Join-Path $r2Workspace '.git\lfs\objects'))
$r2Temporary = Join-Path $r2Workspace 'tmp\r2-lfs-windows'
New-Item -ItemType Directory -Path $r2Temporary -Force | Out-Null
$r2Results = @($r2Batch.objects | ForEach-Object -ThrottleLimit $Throttle -Parallel {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'
    $r2Object = $_
    $r2Oid = [string]$r2Object.oid
    $r2CacheRoot = $using:r2Cache
    $r2TemporaryRoot = $using:r2Temporary
    $r2ConnectionTimeout = $using:ConnectionTimeout
    $r2OperationTimeout = $using:OperationTimeout
    $r2Directory = Join-Path (Join-Path $r2CacheRoot $r2Oid.Substring(0,2)) $r2Oid.Substring(2,2)
    $r2Destination = Join-Path $r2Directory $r2Oid
    $r2Previous = Get-ChildItem -LiteralPath $r2TemporaryRoot -Filter ($r2Oid + '.*.download') |
        Where-Object { $_.Length -gt 0 -and $_.Length -lt [long]$r2Object.size } |
        Sort-Object Length -Descending | Select-Object -First 1
    $r2Partial = if ($r2Previous) { $r2Previous.FullName } else {
        Join-Path $r2TemporaryRoot ($r2Oid + '.' + [guid]::NewGuid().ToString('N') + '.download')
    }
    try {
        if (Test-Path -LiteralPath $r2Destination) {
            if ((Get-Item -LiteralPath $r2Destination).Length -ne [long]$r2Object.size -or
                (Get-FileHash -LiteralPath $r2Destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $r2Oid) {
                throw 'Existing cache object failed validation.'
            }
        } else {
            $r2Headers = @{}
            foreach ($r2Property in $r2Object.actions.download.header.PSObject.Properties) {
                $r2Headers[$r2Property.Name] = [string]$r2Property.Value
            }
            if ($null -eq $script:r2DownloadSession) {
                $script:r2DownloadSession = [Microsoft.PowerShell.Commands.WebRequestSession]::new()
            }
            for ($r2Attempt=0; $r2Attempt -lt 2; $r2Attempt++) {
                try {
                    Invoke-WebRequest -Uri $r2Object.actions.download.href -Headers $r2Headers `
                        -WebSession $script:r2DownloadSession -OutFile $r2Partial -Resume `
                        -ConnectionTimeoutSeconds $r2ConnectionTimeout -OperationTimeoutSeconds $r2OperationTimeout
                    break
                } catch {
                    if ($r2Attempt -eq 1) { throw }
                    Write-Host ("Retry {0}: {1}" -f $r2Oid.Substring(0,12), $_.Exception.GetType().Name)
                }
            }
            if ((Get-Item -LiteralPath $r2Partial).Length -ne [long]$r2Object.size -or
                (Get-FileHash -LiteralPath $r2Partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $r2Oid) {
                throw 'Downloaded object failed validation.'
            }
            New-Item -ItemType Directory -Path $r2Directory -Force | Out-Null
            Move-Item -LiteralPath $r2Partial -Destination $r2Destination
        }
        Write-Host ("Verified {0} ({1:N0} bytes)" -f $r2Oid.Substring(0,12), $r2Object.size)
        [pscustomobject]@{oid=$r2Oid;bytes=[long]$r2Object.size;status='verified'}
    } catch {
        # Download URLs can contain temporary credentials; keep them out of logs.
        Write-Host ("Failed {0}: {1}" -f $r2Oid.Substring(0,12), $_.Exception.GetType().Name)
        [pscustomobject]@{oid=$r2Oid;bytes=[long]$r2Object.size;status='failed';error_type=$_.Exception.GetType().Name}
    }
})
ConvertTo-Json -InputObject $r2Results -Depth 4 | Set-Content -LiteralPath $ResultReport -Encoding utf8
$r2Failures = @($r2Results | Where-Object status -ne 'verified')
Write-Output ("Objects verified: {0}/{1}; failures: {2}" -f ($r2Results.Count-$r2Failures.Count),$r2Results.Count,$r2Failures.Count)
if ($r2Failures.Count) { exit 2 }
