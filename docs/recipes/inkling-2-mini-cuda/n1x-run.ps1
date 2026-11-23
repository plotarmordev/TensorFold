# Inkling 2 Mini (revision ed40e7da62) whole on an RTX Spark N1X laptop, native Windows ARM64, run by its Codync bot:
#   powershell -ExecutionPolicy Bypass -File docs\recipes\inkling-2-mini-cuda\n1x-run.ps1 -Step fatbins|build|check|control|serve
# From a clean checkout with `vcvarsall arm64` set up and CUDA_PATH naming the windows-arm64 toolkit. Installs, downloads
# and deletes nothing outside .\zig-out and $env:TEMP\inkling2-n1x. Each step stops itself inside 10 minutes.
param(
    [Parameter(Mandatory)][ValidateSet('fatbins', 'build', 'check', 'control', 'serve')][string]$Step,
    [string]$Model = 'C:\models\Inkling-2-Mini-ed40e7da62'
)
$ErrorActionPreference = 'Stop'
$out = Join-Path $env:TEMP 'inkling2-n1x'
New-Item -ItemType Directory -Force $out | Out-Null

function Run([string]$exe, [string[]]$argv) {
    Write-Output "> $exe $argv"
    & $exe @argv
    Write-Output "EXIT=$LASTEXITCODE"
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}

# nvidia-smi on the laptop also lists an NVIDIA NPU, so every query names the N1X by its index, found by name
$gpus = nvidia-smi --query-gpu=index,name --format=csv,noheader
$gpu = @($gpus | Where-Object { $_ -match 'N1X' })
if ($gpu.Count -ne 1) { throw "want one N1X in nvidia-smi, got: $($gpus -join '; ')" }
$gpu = $gpu[0].Split(',')[0].Trim()

# The WDDM device pool the driver gives CUDA (the N1X shares it with the CPU; firmware sets its size), MiB
function Pool([string]$label) {
    Write-Output "$label`: $(nvidia-smi -i $gpu --query-gpu=memory.used,memory.total --format=csv,noheader)"
}

Run (Join-Path $env:CUDA_PATH 'bin\nvcc.exe') @('--version')
nvidia-smi -i $gpu --query-gpu=name,driver_version,power.limit,clocks.max.sm --format=csv,noheader
Write-Output "power plan: $(powercfg /getactivescheme)"
$test = 'zig-out\bin\tf-cuda-test.exe'
$server = 'zig-out\native\bin\tensorfold-native.exe'

# A cold build ran 15 min in one piece (run 2), so it is two steps: nvcc's 147 kernel images, then the Zig products over
# them from the cache. Each stops at 9 min; a step cut off resumes from the cache when it runs again.
if ($Step -in 'fatbins', 'build') {
    $targets = if ($Step -eq 'fatbins') { @('fatbins') } else { @('install', 'native') }
    $argv = $targets + @('-Dsm=121', "`"-Dnvcc=$env:CUDA_PATH\bin\nvcc.exe`"", '-Doptimize=ReleaseFast')
    Write-Output "> zig build $argv"
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $zig = Start-Process -FilePath zig -ArgumentList (@('build') + $argv) -PassThru -NoNewWindow
    $null = $zig.Handle # without a handle taken now, .NET Framework loses the exit code once the process ends
    if (-not $zig.WaitForExit(540000)) {
        & taskkill /T /F /PID $zig.Id | Out-Null
        Write-Output "EXIT=timeout after 9 min: run -Step $Step again to resume from the cache"
        exit 1
    }
    Write-Output ("EXIT={0} in {1:N0} s" -f $zig.ExitCode, $clock.Elapsed.TotalSeconds)
    exit $zig.ExitCode
}

$text = 'The three laws of thermodynamics, explained for a curious student:'
if ($Step -eq 'check') {
    # 188's GB10 checks as they are: the served kernels, #4387's serving checks, the float64 reference
    Run $test @('info')
    Run $test @('kernel-contract', 'inkling2 loop')
    $r = New-Object System.Random 4387
    (1..4200 | ForEach-Object { $r.Next(1000, 199000) }) -join ' ' | Set-Content -Encoding ascii "$out\ids.txt"
    Run $test @('inkling2-serving', $Model, "$out\ids.txt")
    Run $test @('inkling2-mini-ref', $Model, '32', $text)
    exit 0
}

# The reference's planted control is its own step: the float64 reference takes 226 s on the N1X, twice is over 10 min.
# Alone, any non-zero exit (a bad path, no GPU) would pass it, so it also needs the plant's own differing tokens.
if ($Step -eq 'control') {
    $env:INKLING2_REF_PLANT = '1'
    $ref = Start-Process -FilePath $test -Wait -PassThru -NoNewWindow -RedirectStandardError "$out\control.log" `
        -ArgumentList @('inkling2-mini-ref', "`"$Model`"", '32', "`"$text`"")
    Get-Content "$out\control.log"
    Write-Output "planted control EXIT=$($ref.ExitCode) (must be non-zero)"
    $planted = Select-String -Path "$out\control.log" -Pattern '\(PLANTED: no messages\): [1-9]\d* of 32 greedy tokens differ'
    if (-not $ref.ExitCode -or -not $planted) { exit 1 }
    exit 0
}

# serve: one server, 1 then 4 greedy 256-token replies, timed by the server's own `done` lines (first token, decode):
# run 2's client timed itself, as PowerShell's line reads fell 4x behind the server's output
Pool 'before the server'
$port = 8188
$proc = Start-Process -FilePath $server -PassThru -NoNewWindow -RedirectStandardOutput "$out\server.log" `
    -RedirectStandardError "$out\server.err" -ArgumentList @('serve', "`"$Model`"", '--backend', 'cuda',
    '--host', '127.0.0.1', '--port', $port, '--name', 'inkling-2-mini', '--parallel', '4', '--context', '8192',
    '--no-update-check')
$loaded = [Diagnostics.Stopwatch]::StartNew()
try {
    while ($true) {
        try { Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 5 | Out-Null; break } catch {}
        if ($proc.HasExited -or $loaded.Elapsed.TotalSeconds -gt 240) { throw 'server did not answer /v1/models' }
        Start-Sleep 1
    }
    Write-Output ("RESULT n1x server ready in {0:N1} s" -f $loaded.Elapsed.TotalSeconds)
    Pool 'server loaded'
    # nvidia-smi samples the pool every 500 ms in its own process for the whole run; its largest sample is the peak
    $smi = Start-Process -FilePath nvidia-smi -PassThru -NoNewWindow -RedirectStandardOutput "$out\pool.csv" `
        -ArgumentList @('-i', $gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', '500')
    Add-Type -AssemblyName System.Net.Http
    $http = New-Object System.Net.Http.HttpClient
    $http.Timeout = [TimeSpan]::FromMinutes(4)
    $body = '{"model":"inkling-2-mini","prompt":"Write a short Python function that computes the Fibonacci ' +
        'sequence and explain it.","max_tokens":256,"temperature":0,"ignore_eos":true}'
    foreach ($n in 1, 4) {
        $before = @(Select-String -Path "$out\server.log" -Pattern '\] done ').Count
        $calls = foreach ($i in 1..$n) {
            $http.PostAsync("http://127.0.0.1:$port/v1/completions",
                (New-Object System.Net.Http.StringContent $body, ([Text.Encoding]::UTF8), 'application/json'))
        }
        foreach ($c in $calls) { if (-not $c.Result.IsSuccessStatusCode) { throw "$n streams: $($c.Result.StatusCode)" } }
        $done = @(Select-String -Path "$out\server.log" -Pattern '\] done ' | Select-Object -Skip $before)
        if ($done.Count -ne $n) { throw "$n streams: want $n done lines, got $($done.Count)" }
        $total = 0
        foreach ($d in $done) {
            if ($d.Line -notmatch 'tokens=256 .*ttft=([\d.]+) .*tok/s=([\d.]+)') { throw "not a 256-token reply: $($d.Line)" }
            $total += [double]$Matches[2]
            Write-Output "RESULT n1x $n stream(s): first token $($Matches[1]) s, decode $($Matches[2]) tok/s"
        }
        Write-Output ("RESULT n1x {0} stream(s) total decode {1:N1} tok/s" -f $n, $total)
    }
    $smi.Kill(); $smi.WaitForExit()
    $peak = (Get-Content "$out\pool.csv" | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum
    if (-not $peak) { throw "no samples in $out\pool.csv" }
    Write-Output "RESULT n1x peak GPU memory sampled every 500 ms: $peak MiB"
    Write-Output "RESULT n1x server process peak working set $([int]($proc.PeakWorkingSet64 / 1MB)) MiB"
} finally {
    foreach ($p in $proc, $smi) { if ($p -and -not $p.HasExited) { $p.Kill() } }
    Get-Content "$out\server.err" -Tail 20
}
