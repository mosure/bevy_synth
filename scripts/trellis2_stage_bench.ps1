param(
    [string[]]$Images = @("docs/output_chair_bg_removed.png"),
    [int]$Iterations = 3,
    [ValidateSet("low", "medium", "high")]
    [string]$Quality = "medium",
    [ValidateSet("auto", "cpu", "wgpu", "cuda")]
    [string]$Device = "wgpu",
    [string]$Binary = "target/release/trellis2_run",
    [string]$WeightsRoot = "",
    [switch]$StrictBenchmark = $true,
    [Nullable[int]]$SamplerStepsOverride = $null,
    [Nullable[int]]$SlatDenseResolution = $null,
    [switch]$DisableRuntimeDecoders,
    [Nullable[int]]$DecoderMaxChildrenPerParent = $null,
    [switch]$MonitorResources = $true,
    [int]$GpuSampleMs = 1000,
    [double]$GpuUtilBottleneckThreshold = 85.0,
    [string]$MonitorLogDir = "tmp/trellis2_monitor",
    [switch]$KeepMonitorLogs = $false
)

$ErrorActionPreference = "Stop"

$script:HasNvidiaSmi = $null -ne (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)
$script:StageNames = @("sparse", "shape_slat", "tex_slat", "decode")

function Set-Or-ClearEnv([string]$Name, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
    } else {
        Set-Item "Env:$Name" $Value
    }
}

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
        if ($line -match '^\[ts_ms=(\d+)\s+t\+[0-9.]+s\]\s+burn_trellis:\s+stage\s+([a-z_]+)\s+begin') {
            $ts = [double]$Matches[1]
            $stage = [string]$Matches[2]
            $beginByStage[$stage] = $ts
            continue
        }
        if ($line -match '^\[ts_ms=(\d+)\s+t\+[0-9.]+s\]\s+burn_trellis:\s+stage\s+([a-z_]+)\s+complete') {
            $ts = [double]$Matches[1]
            $stage = [string]$Matches[2]
            if ($beginByStage.ContainsKey($stage)) {
                $start = [double]$beginByStage[$stage]
                $intervals.Add([pscustomobject]@{
                        stage = $stage
                        start_ms = $start
                        end_ms = $ts
                        duration_ms = ($ts - $start)
                    })
                $beginByStage.Remove($stage)
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

function Run-Trellis([string]$ImagePath, [string]$VariantName, [bool]$SkipPbr) {
    if (-not (Test-Path $ImagePath)) {
        throw "Input image does not exist: $ImagePath"
    }
    Set-Or-ClearEnv "TRELLIS2_SKIP_PBR" ($(if ($SkipPbr) { "1" } else { "" }))

    $args = @(
        "--input-image", $ImagePath,
        "--quality", $Quality,
        "--device", $Device
    )
    if (-not [string]::IsNullOrWhiteSpace($WeightsRoot)) {
        $args += @("--weights-root", $WeightsRoot)
    }
    if ($StrictBenchmark) {
        $args += "--strict-benchmark"
    }

    $runId = [guid]::NewGuid().ToString("N")
    $gpuLogPath = Join-Path $MonitorLogDir ("gpu_{0}.csv" -f $runId)
    $gpuProc = $null
    $lines = @()
    $previousStageDebug = [Environment]::GetEnvironmentVariable("TRELLIS2_STAGE_DEBUG", "Process")
    if ($MonitorResources) {
        Set-Or-ClearEnv "TRELLIS2_STAGE_DEBUG" "1"
    }

    try {
        $gpuProc = Start-GpuMonitor -LogPath $gpuLogPath -SampleMs $GpuSampleMs
        $lines = @(& $Binary @args 2>&1 | ForEach-Object { "$_" })
    } finally {
        Stop-GpuMonitor -Proc $gpuProc
        Set-Or-ClearEnv "TRELLIS2_STAGE_DEBUG" $previousStageDebug
    }

    $jsonLine = @($lines | Where-Object { $_ -match '^\{' } | Select-Object -Last 1)
    if ($jsonLine.Count -eq 0) {
        $joined = ($lines -join "`n")
        throw "trellis2_run produced no JSON output for variant '$VariantName' image '$ImagePath'.`n$joined"
    }

    $parsed = $jsonLine[0] | ConvertFrom-Json
    $intervals = Parse-StageIntervals -Lines $lines
    $gpuSamples = Parse-GpuSamples -LogPath $gpuLogPath
    $gpuAnalysis = Measure-GpuUtilizationByStage `
        -Samples $gpuSamples `
        -Intervals $intervals `
        -UtilThreshold $GpuUtilBottleneckThreshold

    if (-not $KeepMonitorLogs) {
        Remove-Item $gpuLogPath -Force -ErrorAction SilentlyContinue
    }

    return [pscustomobject]@{
        parsed = $parsed
        stage_intervals = $intervals
        gpu = $gpuAnalysis
        stage_lines = @($lines | Where-Object { $_ -match '^\[ts_ms=' })
        raw_line_count = $lines.Count
    }
}

if ($SamplerStepsOverride.HasValue -and $SamplerStepsOverride.Value -gt 0) {
    Set-Or-ClearEnv "TRELLIS2_SAMPLER_STEPS_OVERRIDE" "$($SamplerStepsOverride.Value)"
} else {
    Set-Or-ClearEnv "TRELLIS2_SAMPLER_STEPS_OVERRIDE" ""
}
if ($SlatDenseResolution.HasValue -and $SlatDenseResolution.Value -gt 0) {
    Set-Or-ClearEnv "TRELLIS2_SLAT_DENSE_RESOLUTION" "$($SlatDenseResolution.Value)"
} else {
    Set-Or-ClearEnv "TRELLIS2_SLAT_DENSE_RESOLUTION" ""
}
Set-Or-ClearEnv "TRELLIS2_DISABLE_RUNTIME_DECODERS" ($(if ($DisableRuntimeDecoders) { "1" } else { "" }))
if ($DecoderMaxChildrenPerParent.HasValue) {
    Set-Or-ClearEnv "TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT" "$($DecoderMaxChildrenPerParent.Value)"
} else {
    Set-Or-ClearEnv "TRELLIS2_DECODER_MAX_CHILDREN_PER_PARENT" ""
}

$variants = @(
    @{ name = "mesh_only"; skip_pbr = $true },
    @{ name = "full"; skip_pbr = $false }
)

$runs = New-Object System.Collections.Generic.List[object]
$startedAt = Get-Date

foreach ($variant in $variants) {
    for ($iter = 1; $iter -le $Iterations; $iter++) {
        foreach ($img in $Images) {
            $run = Run-Trellis -ImagePath $img -VariantName $variant.name -SkipPbr ([bool]$variant.skip_pbr)
            $result = $run.parsed
            $timings = $result.timings_ms
            $overallGpu = $run.gpu.overall
            $stageGpu = $run.gpu.stages
            $bottlenecks = @($run.gpu.bottlenecks)
            $runs.Add([pscustomobject]@{
                    variant = $variant.name
                    image = $img
                    iteration = $iter
                    elapsed_ms = [double]$result.elapsed_ms
                    sparse_source = [string]$result.sparse_source
                    decode_source = [string]$result.decode_source
                    vertices = [int64]$result.vertices
                    faces = [int64]$result.faces
                    preprocess = [double]$timings.preprocess
                    runtime_setup = [double]$timings.runtime_setup
                    sparse = [double]$timings.sparse
                    shape_slat = [double]$timings.shape_slat
                    tex_slat = [double]$timings.tex_slat
                    decode = [double]$timings.decode
                    decode_shape_decoder = [double]$timings.decode_shape_decoder
                    decode_tex_decoder = [double]$timings.decode_tex_decoder
                    decode_attr_merge = [double]$timings.decode_attr_merge
                    decode_mesh = [double]$timings.decode_mesh
                    decode_pbr = [double]$timings.decode_pbr
                    hook_capture = [double]$timings.hook_capture
                    host_readback_count = [double]$timings.host_readback_count
                    host_readback_elements = [double]$timings.host_readback_elements
                    total = [double]$timings.total
                    gpu_samples = [int]$overallGpu.samples
                    gpu_util_mean = [double]$overallGpu.util_mean
                    gpu_util_min = [double]$overallGpu.util_min
                    gpu_util_max = [double]$overallGpu.util_max
                    gpu_mem_used_mean_mb = [double]$overallGpu.mem_used_mean_mb
                    gpu_mem_used_peak_mb = [double]$overallGpu.mem_used_peak_mb
                    stage_sparse_wall_ms = Get-StageMetricValue $stageGpu "sparse" "duration_ms"
                    stage_shape_slat_wall_ms = Get-StageMetricValue $stageGpu "shape_slat" "duration_ms"
                    stage_tex_slat_wall_ms = Get-StageMetricValue $stageGpu "tex_slat" "duration_ms"
                    stage_decode_wall_ms = Get-StageMetricValue $stageGpu "decode" "duration_ms"
                    gpu_stage_sparse_util_mean = Get-StageMetricValue $stageGpu "sparse" "util_mean"
                    gpu_stage_shape_slat_util_mean = Get-StageMetricValue $stageGpu "shape_slat" "util_mean"
                    gpu_stage_tex_slat_util_mean = Get-StageMetricValue $stageGpu "tex_slat" "util_mean"
                    gpu_stage_decode_util_mean = Get-StageMetricValue $stageGpu "decode" "util_mean"
                    gpu_stage_sparse_bottleneck = Get-StageBottleneckFlag $stageGpu "sparse"
                    gpu_stage_shape_slat_bottleneck = Get-StageBottleneckFlag $stageGpu "shape_slat"
                    gpu_stage_tex_slat_bottleneck = Get-StageBottleneckFlag $stageGpu "tex_slat"
                    gpu_stage_decode_bottleneck = Get-StageBottleneckFlag $stageGpu "decode"
                    gpu_any_bottleneck = $(if ($bottlenecks.Count -gt 0) { 1 } else { 0 })
                    gpu_bottleneck_stages = ($bottlenecks -join ",")
                    stage_intervals = $run.stage_intervals
                    stage_log_line_count = @($run.stage_lines).Count
                })
        }
    }
}

$fields = @(
    "elapsed_ms",
    "preprocess",
    "runtime_setup",
    "sparse",
    "shape_slat",
    "tex_slat",
    "decode",
    "decode_shape_decoder",
    "decode_tex_decoder",
    "decode_attr_merge",
    "decode_mesh",
    "decode_pbr",
    "hook_capture",
    "host_readback_count",
    "host_readback_elements",
    "total",
    "gpu_samples",
    "gpu_util_mean",
    "gpu_util_min",
    "gpu_util_max",
    "gpu_mem_used_mean_mb",
    "gpu_mem_used_peak_mb",
    "stage_sparse_wall_ms",
    "stage_shape_slat_wall_ms",
    "stage_tex_slat_wall_ms",
    "stage_decode_wall_ms",
    "gpu_stage_sparse_util_mean",
    "gpu_stage_shape_slat_util_mean",
    "gpu_stage_tex_slat_util_mean",
    "gpu_stage_decode_util_mean",
    "gpu_stage_sparse_bottleneck",
    "gpu_stage_shape_slat_bottleneck",
    "gpu_stage_tex_slat_bottleneck",
    "gpu_stage_decode_bottleneck",
    "gpu_any_bottleneck"
)

$variantSummary = @{}
foreach ($variant in $variants) {
    $name = [string]$variant.name
    $subset = @($runs | Where-Object { $_.variant -eq $name })
    $metrics = @{}
    foreach ($field in $fields) {
        $metrics[$field] = Summarize-Field -Rows $subset -Field $field
    }

    $stageSummary = @{}
    foreach ($stage in $script:StageNames) {
        $utilField = "gpu_stage_{0}_util_mean" -f $stage
        $wallField = "stage_{0}_wall_ms" -f $stage
        $bottleneckField = "gpu_stage_{0}_bottleneck" -f $stage
        $stageSummary[$stage] = @{
            util = Summarize-Field -Rows $subset -Field $utilField
            wall_ms = Summarize-Field -Rows $subset -Field $wallField
            bottleneck_runs = @($subset | Where-Object { [int]($_.$bottleneckField) -gt 0 }).Count
        }
    }

    $variantSummary[$name] = @{
        runs = $subset.Count
        metrics = $metrics
        stage_gpu = $stageSummary
        any_bottleneck_runs = @($subset | Where-Object { [int]$_.gpu_any_bottleneck -gt 0 }).Count
    }
}

$report = [ordered]@{
    started_at = $startedAt.ToString("o")
    finished_at = (Get-Date).ToString("o")
    config = [ordered]@{
        binary = $Binary
        images = $Images
        iterations = $Iterations
        quality = $Quality
        device = $Device
        weights_root = $WeightsRoot
        strict_benchmark = [bool]$StrictBenchmark
        sampler_steps_override = $SamplerStepsOverride
        slat_dense_resolution = $SlatDenseResolution
        disable_runtime_decoders = [bool]$DisableRuntimeDecoders
        decoder_max_children_per_parent = $DecoderMaxChildrenPerParent
        monitor_resources = [bool]$MonitorResources
        nvidia_smi_available = [bool]$script:HasNvidiaSmi
        gpu_sample_ms = $GpuSampleMs
        gpu_util_bottleneck_threshold = $GpuUtilBottleneckThreshold
        keep_monitor_logs = [bool]$KeepMonitorLogs
        monitor_log_dir = $MonitorLogDir
    }
    variants = $variantSummary
    runs = $runs
}

$report | ConvertTo-Json -Depth 10
