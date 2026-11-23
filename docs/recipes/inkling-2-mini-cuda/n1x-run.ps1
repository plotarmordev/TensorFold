# Inkling 2 Mini (revision ed40e7da62) whole on an RTX Spark N1X laptop, native Windows ARM64, run by its Codync bot:
#   powershell -ExecutionPolicy Bypass -File docs\recipes\inkling-2-mini-cuda\n1x-run.ps1 -Step build|check|serve
# From a clean checkout with `vcvarsall arm64` set up and CUDA_PATH naming the windows-arm64 toolkit. Installs, downloads
# and deletes nothing outside .\zig-out and $env:TEMP\inkling2-n1x. Each step stops itself well inside 10 minutes.
param(
    [Parameter(Mandatory)][ValidateSet('build', 'check', 'serve')][string]$Step,
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

# The WDDM device pool the driver gives CUDA (the N1X shares it with the CPU; firmware sets its size), MiB
function Pool([string]$label) {
    Write-Output "$label`: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
}

Run (Join-Path $env:CUDA_PATH 'bin\nvcc.exe') @('--version')
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm --format=csv,noheader
Write-Output "power plan: $(powercfg /getactivescheme)"
$test = 'zig-out\bin\tf-cuda-test.exe'
$server = 'zig-out\native\bin\tensorfold-native.exe'

if ($Step -eq 'build') {
    Run zig @('build', 'install', 'native', '-Dsm=121', "-Dnvcc=$env:CUDA_PATH\bin\nvcc.exe", '-Doptimize=ReleaseFast')
    exit 0
}

if ($Step -eq 'check') {
    # 188's GB10 checks as they are: the served kernels, #4387's serving checks, the float64 reference
    Run $test @('info')
    Run $test @('kernel-contract', 'inkling2 loop')
    $r = New-Object System.Random 4387
    (1..4200 | ForEach-Object { $r.Next(1000, 199000) }) -join ' ' | Set-Content -Encoding ascii "$out\ids.txt"
    Run $test @('inkling2-serving', $Model, "$out\ids.txt")
    $text = 'The three laws of thermodynamics, explained for a curious student:'
    Run $test @('inkling2-mini-ref', $Model, '32', $text)
    $env:INKLING2_REF_PLANT = '1'
    & $test inkling2-mini-ref $Model 32 $text
    Write-Output "planted control EXIT=$LASTEXITCODE (must be non-zero)"
    Remove-Item Env:INKLING2_REF_PLANT
    if (-not $LASTEXITCODE) { exit 1 }
    exit 0
}

# serve: one server, 1 then 4 greedy streams of 256 tokens, first token and decode per stream, the pool's peak
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
    Add-Type -AssemblyName System.Net.Http
    $http = New-Object System.Net.Http.HttpClient
    $http.Timeout = [TimeSpan]::FromMinutes(4)
    $peak = 0
    foreach ($n in 1, 4) {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $body = '{"model":"inkling-2-mini","prompt":"Write a short Python function that computes the Fibonacci ' +
            'sequence and explain it.","max_tokens":256,"temperature":0,"ignore_eos":true,"stream":true,' +
            '"stream_options":{"include_usage":true}}'
        $calls = foreach ($i in 1..$n) {
            $req = New-Object System.Net.Http.HttpRequestMessage 'Post', "http://127.0.0.1:$port/v1/completions"
            $req.Content = New-Object System.Net.Http.StringContent $body, ([Text.Encoding]::UTF8), 'application/json'
            $http.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead)
        }
        $streams = foreach ($c in $calls) {
            $reader = New-Object IO.StreamReader $c.Result.Content.ReadAsStreamAsync().Result
            @{ reader = $reader; first = $null; last = $null; tokens = $null }
        }
        # read the streams in turn: a reply's pieces wait in its socket, the clock stamps when each is read
        while ($streams | Where-Object { $_.reader }) {
            foreach ($s in $streams | Where-Object { $_.reader }) {
                $line = $s.reader.ReadLine()
                if ($null -eq $line -or $line -eq 'data: [DONE]') { $s.reader = $null; continue }
                if (-not $line.StartsWith('data: ')) { continue }
                $j = $line.Substring(6) | ConvertFrom-Json
                if ($j.usage) { $s.tokens = $j.usage.completion_tokens }
                if ($j.choices -and $j.choices[0].text) {
                    if ($null -eq $s.first) { $s.first = $clock.Elapsed.TotalSeconds }
                    $s.last = $clock.Elapsed.TotalSeconds
                }
            }
            $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
            if ($used -gt $peak) { $peak = $used }
        }
        $total = 0
        foreach ($s in $streams) {
            if ($s.tokens -ne 256) { throw "$n streams: a reply ended at $($s.tokens) tokens" }
            $tps = ($s.tokens - 1) / ($s.last - $s.first)
            $total += $s.tokens - 1
            Write-Output ("RESULT n1x {0} stream(s): first token {1:N3} s, decode {2:N1} tok/s" -f $n, $s.first, $tps)
        }
        $span = ($streams | ForEach-Object { $_.last } | Measure-Object -Maximum).Maximum -
            ($streams | ForEach-Object { $_.first } | Measure-Object -Minimum).Minimum
        Write-Output ("RESULT n1x {0} stream(s) total decode {1:N1} tok/s" -f $n, ($total / $span))
    }
    Write-Output "RESULT n1x peak GPU memory sampled while decoding: $peak MiB"
    Write-Output "RESULT n1x server process peak working set $([int]($proc.PeakWorkingSet64 / 1MB)) MiB"
} finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id }
    Get-Content "$out\server.err" -Tail 20
}
