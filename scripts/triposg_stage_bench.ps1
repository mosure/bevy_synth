param(
    [string[]]$Images = @("docs/input_chair.jpg"),
    [int]$Iterations = 3,
    [ValidateSet("cpu", "wgpu", "cuda")]
    [string]$Backend = "wgpu",
    [ValidateSet("rmbg14", "rmbg2")]
    [string]$RmbgModel = "rmbg14",
    [string]$BurnSynthExe = "target/release/burn_synth.exe",
    [switch]$UseCargoRun = $false,
    [switch]$MonitorResources = $true,
    [int]$GpuSampleMs = 1000,
    [double]$GpuUtilBottleneckThreshold = 85.0,
    [string]$MonitorLogDir = "tmp/triposg_stage_bench",
    [switch]$KeepMonitorLogs = $false
)

$ErrorActionPreference = "Stop"

$script:HasNvidiaSmi = $null -ne (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)

function Get-Percentile([double[]]$Values, [double]$Percentile) {
    if ($null -eq $Values -or $Values.Count -eq 0) { return [double]::NaN }
    $sorted = $Values | Sort-Object
    if ($sorted.Count -eq 1) { return [double]$sorted[0] }
    $rank = ($Percentile / 100.0) * ($sorted.Count - 1)
    $low = [math]::Floor($rank)
    $high = [math]::Ceiling($rank)
    if ($low -eq $high) { return [double]$sorted[$low] }
    $weight = $rank - $low
    return ([double]$sorted[$low] * (1.0 - $weight)) + ([double]$sorted[$high] * $weight)
}

function Summarize-Field($Rows, [string]$Field) {
    $vals = @(
        $Rows |
            ForEach-Object { [double]($_.$Field) } |
            Where-Object { -not [double]::IsNaN($_) -and -not [double]::IsInfinity($_) }
    )
    if ($vals.Count -eq 0) {
        return @{
            count = 0
            mean = [double]::NaN
            median = [double]::NaN
            p95 = [double]::NaN
            min = [double]::NaN
            max = [double]::NaN
        }
    }
    return @{
        count = $vals.Count
        mean = ($vals | Measure-Object -Average).Average
        median = Get-Percentile $vals 50.0
        p95 = Get-Percentile $vals 95.0
        min = ($vals | Measure-Object -Minimum).Minimum
        max = ($vals | Measure-Object -Maximum).Maximum
    }
}

function Parse-IsoUtcToMs([string]$Value) {
    $dto = [datetimeoffset]::MinValue
    if ([datetimeoffset]::TryParse($Value.Trim(), [ref]$dto)) {
        return [double]$dto.ToUnixTimeMilliseconds()
    }
    return [double]::NaN
}

function Try-ParseTimestampMs([string]$Value) {
    $formats = @(
        "yyyy/MM/dd HH:mm:ss.fff",
        "yyyy/MM/dd HH:mm:ss"
    )
    foreach ($format in $formats) {
        $dt = [datetime]::MinValue
        if ([datetime]::TryParseExact(
                $Value.Trim(),
                $format,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [System.Globalization.DateTimeStyles]::AssumeLocal,
                [ref]$dt
            )) {
            return [double]([datetimeoffset]$dt).ToUnixTimeMilliseconds()
        }
    }
    return [double]::NaN
}

function Parse-GpuSamples([string]$LogPath) {
    if (-not (Test-Path $LogPath)) { return @() }
    $lines = Get-Content -Path $LogPath -ErrorAction SilentlyContinue
    $samples = New-Object System.Collections.Generic.List[object]
    foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line -match '^\]0;') { continue }
        $parts = @($line.Split(',') | ForEach-Object { $_.Trim() })
        if ($parts.Count -lt 5) { continue }
        $tsMs = Try-ParseTimestampMs $parts[0]
        if ([double]::IsNaN($tsMs)) { continue }
        $samples.Add([pscustomobject]@{
                ts_ms = $tsMs
                util_gpu = [double]$parts[1]
                util_mem = [double]$parts[2]
                mem_used_mb = [double]$parts[3]
                mem_total_mb = [double]$parts[4]
            })
    }
    return $samples.ToArray()
}

function Parse-StageIntervals([string[]]$Lines) {
    $beginByStage = @{}
    $intervals = New-Object System.Collections.Generic.List[object]
    foreach ($line in $Lines) {
        if (
            $line -match
            '^\[(?<ts>[^\]]+)\s+INFO\s+burn_synth::progress\]\s+burn_synth\.progress run=mesh stage=(?<stage>[^ ]+) status=(?<status>started|completed).*$'
        ) {
            $tsMs = Parse-IsoUtcToMs $Matches["ts"]
            if ([double]::IsNaN($tsMs)) { continue }
            $stage = [string]$Matches["stage"]
            $status = [string]$Matches["status"]
            if ($status -eq "started") {
                $beginByStage[$stage] = $tsMs
                continue
            }
            if ($status -eq "completed" -and $beginByStage.ContainsKey($stage)) {
                $start = [double]$beginByStage[$stage]
                $intervals.Add([pscustomobject]@{
                        stage = $stage
                        start_ms = $start
                        end_ms = $tsMs
                        duration_ms = ($tsMs - $start)
                    })
                $beginByStage.Remove($stage) | Out-Null
            }
        }
    }
    return $intervals.ToArray()
}

function Summarize-GpuWindow($Samples) {
    $count = @($Samples).Count
    if ($count -eq 0) {
        return [pscustomobject]@{
            samples = 0
            util_mean = [double]::NaN
            util_min = [double]::NaN
            util_max = [double]::NaN
            mem_used_mean_mb = [double]::NaN
            mem_used_peak_mb = [double]::NaN
        }
    }
    $utils = @($Samples | ForEach-Object { [double]$_.util_gpu })
    $mem = @($Samples | ForEach-Object { [double]$_.mem_used_mb })
    return [pscustomobject]@{
        samples = $count
        util_mean = ($utils | Measure-Object -Average).Average
        util_min = ($utils | Measure-Object -Minimum).Minimum
        util_max = ($utils | Measure-Object -Maximum).Maximum
        mem_used_mean_mb = ($mem | Measure-Object -Average).Average
        mem_used_peak_mb = ($mem | Measure-Object -Maximum).Maximum
    }
}

function Measure-GpuUtilizationByStage($Samples, $Intervals, [double]$UtilThreshold) {
    $stageMetrics = @{}
    $bottlenecks = New-Object System.Collections.Generic.List[string]
    foreach ($interval in $Intervals) {
        $start = [double]$interval.start_ms
        $end = [double]$interval.end_ms
        $window = @($Samples | Where-Object { $_.ts_ms -ge $start -and $_.ts_ms -le $end })
        $summary = Summarize-GpuWindow $window
        $isBottleneck =
            ($summary.samples -gt 0) -and
            (-not [double]::IsNaN($summary.util_mean)) -and
            ($summary.util_mean -lt $UtilThreshold)
        if ($isBottleneck) {
            [void]$bottlenecks.Add([string]$interval.stage)
        }
        $stageMetrics[[string]$interval.stage] = [pscustomobject]@{
            start_ms = $start
            end_ms = $end
            duration_ms = [double]$interval.duration_ms
            samples = [int]$summary.samples
            util_mean = [double]$summary.util_mean
            util_min = [double]$summary.util_min
            util_max = [double]$summary.util_max
            mem_used_mean_mb = [double]$summary.mem_used_mean_mb
            mem_used_peak_mb = [double]$summary.mem_used_peak_mb
            bottleneck = [bool]$isBottleneck
        }
    }

    $overall = Summarize-GpuWindow $Samples
    return [pscustomobject]@{
        overall = $overall
        stages = $stageMetrics
        bottlenecks = @($bottlenecks)
    }
}

function Get-StageMetricValue($StageMetrics, [string]$Stage, [string]$Field) {
    if ($null -eq $StageMetrics) { return [double]::NaN }
    if (-not $StageMetrics.ContainsKey($Stage)) { return [double]::NaN }
    $value = $StageMetrics[$Stage].$Field
    if ($null -eq $value) { return [double]::NaN }
    return [double]$value
}

function Get-StageBottleneckFlag($StageMetrics, [string]$Stage) {
    if ($null -eq $StageMetrics) { return 0 }
    if (-not $StageMetrics.ContainsKey($Stage)) { return 0 }
    if ($StageMetrics[$Stage].bottleneck) { return 1 }
    return 0
}

function Start-GpuMonitor([string]$LogPath, [int]$SampleMs) {
    if (-not $MonitorResources -or -not $script:HasNvidiaSmi) {
        return $null
    }
    New-Item -ItemType Directory -Path (Split-Path -Path $LogPath -Parent) -Force | Out-Null
    if (Test-Path $LogPath) {
        Remove-Item $LogPath -Force -ErrorAction SilentlyContinue
    }
    $args = "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits -lms $SampleMs"
    return Start-Process -FilePath "nvidia-smi" -ArgumentList $args -RedirectStandardOutput $LogPath -WindowStyle Hidden -PassThru
}

function Stop-GpuMonitor($Proc) {
    if ($null -eq $Proc) { return }
    if ($Proc.HasExited) { return }
    Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 100
}

function Invoke-BurnSynthMesh([string]$ImagePath, [string]$OutputPath) {
    if (-not (Test-Path $ImagePath)) {
        throw "Input image does not exist: $ImagePath"
    }
    $args = @(
        "--synthesis-models", "triposg",
        "--rmbg-model", $RmbgModel,
        "--backend", $Backend,
        "--progress", "stages",
        "mesh",
        "--input", $ImagePath,
        "--output", $OutputPath
    )

    if ($UseCargoRun) {
        $features = @("runtime")
        if ($Backend -eq "wgpu") {
            $features += "wgpu"
        } elseif ($Backend -eq "cuda") {
            $features += "cuda"
        }
        return @(& cargo run -p burn_synth --features ($features -join ",") -- @args 2>&1 | ForEach-Object { "$_" })
    }
    if (-not (Test-Path $BurnSynthExe)) {
        throw "burn_synth executable not found at '$BurnSynthExe'. Build it first or pass -UseCargoRun."
    }
    return @(& $BurnSynthExe @args 2>&1 | ForEach-Object { "$_" })
}

if ($Iterations -lt 1) {
    throw "Iterations must be >= 1."
}
if ($MonitorResources -and -not $script:HasNvidiaSmi) {
    Write-Warning "nvidia-smi not found; resource monitoring disabled."
    $MonitorResources = $false
}
New-Item -ItemType Directory -Path $MonitorLogDir -Force | Out-Null

$rows = @()
for ($iter = 1; $iter -le $Iterations; $iter++) {
    foreach ($image in $Images) {
        $slug = [System.IO.Path]::GetFileNameWithoutExtension($image)
        $runId = [guid]::NewGuid().ToString("N")
        $gpuLogPath = Join-Path $MonitorLogDir ("gpu_{0}.csv" -f $runId)
        $outPath = Join-Path $MonitorLogDir ("{0}_{1}.glb" -f $slug, $runId)

        $gpuProc = $null
        $lines = @()
        try {
            $gpuProc = Start-GpuMonitor -LogPath $gpuLogPath -SampleMs $GpuSampleMs
            $lines = Invoke-BurnSynthMesh -ImagePath $image -OutputPath $outPath
        } finally {
            Stop-GpuMonitor -Proc $gpuProc
        }

        $completedLine = @($lines | Where-Object { $_ -match 'run=mesh status=completed elapsed_ms=' } | Select-Object -Last 1)
        if ($completedLine.Count -eq 0) {
            $joined = ($lines -join "`n")
            throw "burn_synth did not emit completion line for image '$image'.`n$joined"
        }
        $elapsedMatch = [regex]::Match($completedLine[0], 'elapsed_ms=([0-9\.]+)')
        $totalMs = if ($elapsedMatch.Success) { [double]$elapsedMatch.Groups[1].Value } else { [double]::NaN }

        $intervals = Parse-StageIntervals -Lines $lines
        $gpuSamples = Parse-GpuSamples -LogPath $gpuLogPath
        $gpuAnalysis = Measure-GpuUtilizationByStage `
            -Samples $gpuSamples `
            -Intervals $intervals `
            -UtilThreshold $GpuUtilBottleneckThreshold

        if (-not $KeepMonitorLogs) {
            Remove-Item $gpuLogPath -Force -ErrorAction SilentlyContinue
            Remove-Item $outPath -Force -ErrorAction SilentlyContinue
        }

        $stageMetrics = $gpuAnalysis.stages
        $rows += [pscustomobject]@{
            iter = $iter
            image = $image
            total_ms = $totalMs
            preprocess_ms = Get-StageMetricValue $stageMetrics "mesh.preprocess_foreground" "duration_ms"
            load_ms = Get-StageMetricValue $stageMetrics "triposg.load_backend" "duration_ms"
            encode_ms = Get-StageMetricValue $stageMetrics "triposg.encode_image" "duration_ms"
            sample_ms = Get-StageMetricValue $stageMetrics "triposg.sample" "duration_ms"
            flash_extract_ms = Get-StageMetricValue $stageMetrics "triposg.flash_extract" "duration_ms"
            mesh_extract_ms = Get-StageMetricValue $stageMetrics "triposg.mesh_extract" "duration_ms"
            decimate_ms = Get-StageMetricValue $stageMetrics "mesh.decimate" "duration_ms"
            gpu_util_mean = [double]$gpuAnalysis.overall.util_mean
            gpu_util_p95 = Get-Percentile @($gpuSamples | ForEach-Object { [double]$_.util_gpu }) 95.0
            gpu_mem_peak_mb = [double]$gpuAnalysis.overall.mem_used_peak_mb
            sample_stage_gpu_util_mean = Get-StageMetricValue $stageMetrics "triposg.sample" "util_mean"
            flash_stage_gpu_util_mean = Get-StageMetricValue $stageMetrics "triposg.flash_extract" "util_mean"
            sample_stage_bottleneck = Get-StageBottleneckFlag $stageMetrics "triposg.sample"
            flash_stage_bottleneck = Get-StageBottleneckFlag $stageMetrics "triposg.flash_extract"
            mesh_extract_stage_bottleneck = Get-StageBottleneckFlag $stageMetrics "triposg.mesh_extract"
        }
    }
}

$summary = [ordered]@{
    total_ms = Summarize-Field $rows "total_ms"
    preprocess_ms = Summarize-Field $rows "preprocess_ms"
    load_ms = Summarize-Field $rows "load_ms"
    encode_ms = Summarize-Field $rows "encode_ms"
    sample_ms = Summarize-Field $rows "sample_ms"
    flash_extract_ms = Summarize-Field $rows "flash_extract_ms"
    mesh_extract_ms = Summarize-Field $rows "mesh_extract_ms"
    decimate_ms = Summarize-Field $rows "decimate_ms"
    gpu_util_mean = Summarize-Field $rows "gpu_util_mean"
    gpu_util_p95 = Summarize-Field $rows "gpu_util_p95"
    gpu_mem_peak_mb = Summarize-Field $rows "gpu_mem_peak_mb"
    sample_stage_gpu_util_mean = Summarize-Field $rows "sample_stage_gpu_util_mean"
    flash_stage_gpu_util_mean = Summarize-Field $rows "flash_stage_gpu_util_mean"
    sample_stage_bottleneck_rate = (($rows | Measure-Object -Property sample_stage_bottleneck -Average).Average)
    flash_stage_bottleneck_rate = (($rows | Measure-Object -Property flash_stage_bottleneck -Average).Average)
    mesh_extract_stage_bottleneck_rate = (($rows | Measure-Object -Property mesh_extract_stage_bottleneck -Average).Average)
}

$report = [ordered]@{
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    backend = $Backend
    rmbg_model = $RmbgModel
    monitor_resources = [bool]$MonitorResources
    gpu_sample_ms = $GpuSampleMs
    bottleneck_threshold = $GpuUtilBottleneckThreshold
    rows = $rows
    summary = $summary
}

$timestamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$reportPath = Join-Path $MonitorLogDir "triposg_stage_bench_$timestamp.json"
$report | ConvertTo-Json -Depth 8 | Set-Content -Path $reportPath -Encoding UTF8

Write-Host "Triposg stage bench report: $reportPath"
Write-Host ""
Write-Host "Summary:"
$summary | ConvertTo-Json -Depth 6
