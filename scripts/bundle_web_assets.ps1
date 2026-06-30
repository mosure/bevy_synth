param(
    [string]$DestinationRoot = "www/assets",
    [switch]$NoClean,
    [switch]$DryRun,
    [switch]$ExcludeRmbg2,
    [switch]$AllowMissingTrellis
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    $scriptRoot = $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($scriptRoot) -and -not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
        $scriptRoot = Split-Path -Parent $PSCommandPath
    }
    if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
        throw "Unable to resolve script root."
    }
    return (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

function Resolve-PathFromRepo([string]$repoRoot, [string]$path) {
    if ([System.IO.Path]::IsPathRooted($path)) {
        return $path
    }
    return Join-Path $repoRoot $path
}

function Get-DirectoryStats([string]$path) {
    if (-not (Test-Path $path)) {
        return [pscustomobject]@{
            FileCount = 0
            TotalBytes = 0
        }
    }

    $files = Get-ChildItem -Recurse -File $path
    $measure = $files | Measure-Object -Property Length -Sum
    return [pscustomobject]@{
        FileCount = [int]$measure.Count
        TotalBytes = [int64]$measure.Sum
    }
}

function Test-AllGlobMatch([string]$source, [string[]]$globs) {
    $missing = New-Object System.Collections.Generic.List[string]
    foreach ($glob in $globs) {
        $matches = Get-ChildItem -Path (Join-Path $source $glob) -File -ErrorAction SilentlyContinue
        if (-not $matches) {
            $missing.Add($glob)
        }
    }
    return [pscustomobject]@{
        Ok = ($missing.Count -eq 0)
        Missing = $missing
    }
}

function Copy-RuntimeFiles(
    [string]$source,
    [string]$destination,
    [bool]$dryRun,
    [string[]]$copyGlobs
) {
    if (-not $copyGlobs -or $copyGlobs.Count -eq 0) {
        $copyGlobs = @("*.bpk")
    }

    $files = Get-ChildItem -Recurse -File $source | Where-Object {
        $relative = [System.IO.Path]::GetRelativePath($source, $_.FullName).Replace('\', '/')
        if ($relative -like "*.bpk.meta.json") {
            return $false
        }
        foreach ($glob in $copyGlobs) {
            if ($relative -like $glob) {
                return $true
            }
        }
        return $false
    }

    if ($dryRun) {
        Write-Host ("[DRY RUN] Copy {0} filtered files from {1} -> {2}" -f $files.Count, $source, $destination)
        return $files.Count
    }

    if (Test-Path $destination) {
        Remove-Item -Recurse -Force $destination
    }
    New-Item -ItemType Directory -Path $destination -Force | Out-Null

    foreach ($file in $files) {
        $relative = [System.IO.Path]::GetRelativePath($source, $file.FullName)
        $target = Join-Path $destination $relative
        $targetParent = Split-Path -Parent $target
        if (-not (Test-Path $targetParent)) {
            New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
        }
        Copy-Item -Path $file.FullName -Destination $target -Force
    }

    return $files.Count
}

function Ensure-FileAlias {
    param(
        [string]$Root,
        [string]$TargetRelative,
        [string[]]$SourceRelatives,
        [bool]$DryRun
    )

    $targetPath = Join-Path $Root $TargetRelative
    if (Test-Path $targetPath) {
        return $false
    }

    foreach ($sourceRelative in $SourceRelatives) {
        $sourcePath = Join-Path $Root $sourceRelative
        if (-not (Test-Path $sourcePath)) {
            continue
        }

        if ($DryRun) {
            Write-Host ("[DRY RUN] Alias missing metadata: {0} -> {1}" -f $sourceRelative, $TargetRelative)
            return $true
        }

        $targetParent = Split-Path -Parent $targetPath
        if (-not (Test-Path $targetParent)) {
            New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
        }
        Copy-Item -Path $sourcePath -Destination $targetPath -Force
        Write-Host ("Created metadata alias: {0} -> {1}" -f $sourceRelative, $TargetRelative)
        return $true
    }

    return $false
}

function Ensure-TripoMetadataAliases {
    param(
        [string]$ModelRoot,
        [bool]$DryRun
    )

    $created = 0
    if (Ensure-FileAlias -Root $ModelRoot -TargetRelative "image_encoder_dinov2/config.json" -SourceRelatives @("image_encoder_2/config.json", "image_encoder_1/config.json") -DryRun:$DryRun) {
        $created += 1
    }
    if (Ensure-FileAlias -Root $ModelRoot -TargetRelative "feature_extractor_dinov2/preprocessor_config.json" -SourceRelatives @("feature_extractor_2/preprocessor_config.json", "feature_extractor_1/preprocessor_config.json") -DryRun:$DryRun) {
        $created += 1
    }
    return $created
}

$repoRoot = Resolve-RepoRoot
$destinationRootAbs = Resolve-PathFromRepo -repoRoot $repoRoot -path $DestinationRoot
$modelsRoot = Join-Path $destinationRootAbs "models"

$sources = @(
    @{
        Name = "MIDI-3D"
        Source = Join-Path $repoRoot "crates/burn_tripo/assets/models/MIDI-3D"
        Required = $true
        RequiredGlobs = @(
            "vae/diffusion_pytorch_model.bpk",
            "vae/diffusion_pytorch_model.bpk.parts.json",
            "vae/diffusion_pytorch_model_f16.bpk",
            "vae/diffusion_pytorch_model_f16.bpk.parts.json",
            "transformer/diffusion_pytorch_model.bpk",
            "transformer/diffusion_pytorch_model.bpk.parts.json",
            "transformer/diffusion_pytorch_model_f16.bpk",
            "transformer/diffusion_pytorch_model_f16.bpk.parts.json",
            "image_encoder_dinov2/model.bpk",
            "image_encoder_dinov2/model.bpk.parts.json",
            "image_encoder_dinov2/model_f16.bpk",
            "image_encoder_dinov2/model_f16.bpk.parts.json"
        )
        CopyGlobs = @(
            "*.bpk.parts.json",
            "*.bpk.part-*",
            "vae/config.json",
            "transformer/config.json",
            "scheduler/scheduler_config.json",
            "image_encoder_dinov2/config.json",
            "image_encoder_2/config.json",
            "image_encoder_1/config.json",
            "feature_extractor_dinov2/preprocessor_config.json",
            "feature_extractor_2/preprocessor_config.json",
            "feature_extractor_1/preprocessor_config.json"
        )
    },
    @{
        Name = "RMBG-1.4"
        Source = Join-Path $repoRoot "crates/burn_foreground/assets/models/RMBG-1.4"
        Required = $true
        RequiredGlobs = @("model*.bpk", "model*.bpk.parts.json")
        CopyGlobs = @(
            "*.bpk.parts.json",
            "*.bpk.part-*",
            "config.json",
            "preprocessor_config.json"
        )
    },
    @{
        Name = "RMBG-2.0"
        Source = Join-Path $repoRoot "crates/burn_foreground/assets/models/RMBG-2.0"
        Required = $true
        RequiredGlobs = @("model*.bpk", "model*.bpk.parts.json")
        CopyGlobs = @(
            "*.bpk.parts.json",
            "*.bpk.part-*",
            "config.json",
            "preprocessor_config.json"
        )
    },
    @{
        Name = "TRELLIS.2-4B"
        Source = Join-Path $repoRoot "crates/burn_trellis/assets/models/TRELLIS.2-4B"
        Required = -not $AllowMissingTrellis
        RequiredGlobs = @(
            "pipeline.json",
            "ckpts/*.bpk",
            "ckpts/*.bpk.parts.json",
            "facebook/dinov3-vitl16-pretrain-lvd1689m/model.bpk",
            "facebook/dinov3-vitl16-pretrain-lvd1689m/model.bpk.parts.json",
            "facebook/dinov3-vitl16-pretrain-lvd1689m/model_f16.bpk",
            "facebook/dinov3-vitl16-pretrain-lvd1689m/model_f16.bpk.parts.json"
        )
        CopyGlobs = @(
            "*.bpk.parts.json",
            "*.bpk.part-*",
            "pipeline.json",
            "ckpts/*.json",
            "facebook/*/config.json"
        )
    },
    @{
        Name = "TRELLIS-image-large"
        Source = Join-Path $repoRoot "crates/burn_trellis/assets/models/TRELLIS-image-large"
        Required = -not $AllowMissingTrellis
        RequiredGlobs = @(
            "ckpts/*.bpk",
            "ckpts/*.bpk.parts.json"
        )
        CopyGlobs = @(
            "*.bpk.parts.json",
            "*.bpk.part-*",
            "ckpts/*.json"
        )
    }
)

Write-Host "Repo root: $repoRoot"
Write-Host "Destination root: $destinationRootAbs"

if (-not $NoClean -and (Test-Path $modelsRoot)) {
    if ($DryRun) {
        Write-Host "[DRY RUN] Remove existing model tree: $modelsRoot"
    } else {
        Write-Host "Removing existing model tree: $modelsRoot"
        Remove-Item -Recurse -Force $modelsRoot
    }
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Path $modelsRoot -Force | Out-Null
}

$rows = New-Object System.Collections.Generic.List[object]

foreach ($entry in $sources) {
    $name = $entry.Name
    $source = $entry.Source
    $required = [bool]$entry.Required
    $requiredGlobs = $entry.RequiredGlobs
    $copyGlobs = $entry.CopyGlobs
    $destination = Join-Path $modelsRoot $name

    if ($name -eq "RMBG-2.0" -and $ExcludeRmbg2) {
        Write-Host "Skipping '$name' due to -ExcludeRmbg2."
        continue
    }

    if (-not (Test-Path $source)) {
        if ($required) {
            throw "Required source is missing: $source"
        }
        Write-Host "Skipping optional source '$name' (not found): $source"
        continue
    }

    $requiredCheck = Test-AllGlobMatch -source $source -globs $requiredGlobs
    if (-not $requiredCheck.Ok) {
        $missingPatterns = ($requiredCheck.Missing -join ", ")
        if ($required) {
            throw "Required model '$name' is missing expected burnpack files under: $source (missing patterns: $missingPatterns)"
        }
        Write-Host "Skipping optional source '$name' (missing expected burnpack files: $missingPatterns): $source"
        continue
    }

    Write-Host "Bundling '$name'..."
    Copy-RuntimeFiles -source $source -destination $destination -dryRun:$DryRun -copyGlobs $copyGlobs | Out-Null

    if ($DryRun) {
        $rows.Add([pscustomobject]@{
                Model = $name
                Source = $source
                Destination = $destination
                Files = "-"
                Bytes = "-"
            })
    } else {
        $stats = Get-DirectoryStats -path $destination
        $rows.Add([pscustomobject]@{
                Model = $name
                Source = $source
                Destination = $destination
                Files = $stats.FileCount
                Bytes = $stats.TotalBytes
            })
    }
}

if (Test-Path (Join-Path $modelsRoot "MIDI-3D")) {
    Ensure-TripoMetadataAliases -ModelRoot (Join-Path $modelsRoot "MIDI-3D") -DryRun:$DryRun | Out-Null
}

Write-Host ""
Write-Host "Bundle summary:"
$rows | Format-Table -AutoSize

if (-not $DryRun) {
    $totals = Get-DirectoryStats -path $modelsRoot
    Write-Host ""
    Write-Host ("Bundled {0} files ({1} bytes) into {2}" -f $totals.FileCount, $totals.TotalBytes, $modelsRoot)
}
