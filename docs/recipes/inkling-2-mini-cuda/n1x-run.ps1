# Inkling 2 Mini (revision ed40e7da62) whole on an RTX Spark N1X laptop, native Windows ARM64, run by its Codync bot:
#   powershell -ExecutionPolicy Bypass -File docs\recipes\inkling-2-mini-cuda\n1x-run.ps1 -Step fatbins|images|runner|server|check|control|serve
# From a clean checkout with `vcvarsall arm64` set up and CUDA_PATH naming the windows-arm64 toolkit. Installs, downloads
# and deletes nothing outside .\zig-out and $env:TEMP\inkling2-n1x. Each step stops itself inside 10 minutes.
# Card 257 pairs our plain-decode build with Mia's DGX Spark recipe (README.md); it serves at --parallel 8 and measures
# 1, 2, 4 and 8 greedy streams, a prefill probe, output agreement and peak WDDM pool memory against her published rows.
param(
    [Parameter(Mandatory)][ValidateSet('fatbins', 'images', 'runner', 'server', 'check', 'control', 'serve')][string]$Step,
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

# A cold build ran 15 min in one piece (run 2) and `install native` twice ran past 9 min (run 4), building a dozen
# products this card does not run, so each step builds one thing: nvcc's 143 kernel images, the 37 contract images
# tf-cuda-test embeds (DeepSeek's EXL3 units, controls, H3: run 5's runner step ran nvcc on them and stopped twice),
# then tf-cuda-test, then tensorfold-native. Each stops at 9 min; a step cut off resumes from the cache when it runs
# again, and its log's tail and the images installed so far show how far it got.
$targets = @{ fatbins = 'fatbins'; images = 'kernel-contract-fatbins'; runner = 'tf-cuda-test'; server = 'native' }
if ($targets.ContainsKey($Step)) {
    $argv = @($targets[$Step], '-Dsm=121', "`"-Dnvcc=$env:CUDA_PATH\bin\nvcc.exe`"", '-Doptimize=ReleaseFast')
    Write-Output "> zig build $argv"
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $zig = Start-Process -FilePath zig -ArgumentList (@('build') + $argv) -PassThru -NoNewWindow `
        -RedirectStandardOutput "$out\$Step.out" -RedirectStandardError "$out\$Step.log"
    $null = $zig.Handle # without a handle taken now, .NET Framework loses the exit code once the process ends
    $done = $zig.WaitForExit(540000)
    if (-not $done) { & taskkill /T /F /PID $zig.Id | Out-Null; $zig.WaitForExit() }
    Get-Content "$out\$Step.out", "$out\$Step.log" -Tail 30
    Write-Output "zig-out\fatbin: $(@(Get-ChildItem zig-out\fatbin -Filter *.fatbin -ErrorAction SilentlyContinue).Count) images"
    if (-not $done) {
        Write-Output "EXIT=timeout after 9 min: run -Step $Step again to resume from the cache"
        exit 1
    }
    Write-Output ("EXIT={0} in {1:N0} s" -f $zig.ExitCode, $clock.Elapsed.TotalSeconds)
    exit $zig.ExitCode
}

# check, control and serve share the same 9-minute deadline the build steps enforce, so a stalled check, reference
# or request cannot run past the step's stated 10-minute bound
$clock = [Diagnostics.Stopwatch]::StartNew()
function Left() { [Math]::Max(1, 540000 - [int]$clock.ElapsedMilliseconds) }
# One tf-cuda-test call under the deadline, its output and error logged like a build step's. -FailOk hands the exit
# code back in $script:last instead of stopping, for the control that must exit non-zero.
function Timed([string[]]$argv, [string]$log, [switch]$FailOk) {
    Write-Output "> $test $argv"
    $p = Start-Process -FilePath $test -PassThru -NoNewWindow -RedirectStandardOutput "$out\$log.out" `
        -RedirectStandardError "$out\$log.err" `
        -ArgumentList @(($argv | ForEach-Object { if ($_ -match ' ') { "`"$_`"" } else { $_ } }))
    $null = $p.Handle # as with the build steps, the handle must be taken for the exit code
    $done = $p.WaitForExit((Left))
    if (-not $done) { & taskkill /T /F /PID $p.Id | Out-Null; $p.WaitForExit() }
    Get-Content "$out\$log.out", "$out\$log.err" -Tail 30 -ErrorAction SilentlyContinue
    $script:last = if ($done) { $p.ExitCode } else { 124 }
    Write-Output "EXIT=$script:last$(if (-not $done) { ' (timeout: the step deadline passed)' })"
    if ($script:last -and -not $FailOk) { exit $script:last }
}

$text = 'The three laws of thermodynamics, explained for a curious student:'
if ($Step -eq 'check') {
    # 188's GB10 checks as they are: the served kernels, #4387's serving checks, the float64 reference. Its sampled
    # case=1 is a known engine defect (card 263), so its exit is returned last and the reference still runs.
    Timed @('info') 'info'
    Timed @('kernel-contract', 'inkling2 loop') 'contract'
    $r = New-Object System.Random 4387
    (1..4200 | ForEach-Object { $r.Next(1000, 199000) }) -join ' ' | Set-Content -Encoding ascii "$out\ids.txt"
    Timed @('inkling2-serving', $Model, "$out\ids.txt") 'serving' -FailOk
    $serving = $script:last
    Timed @('inkling2-mini-ref', $Model, '32', $text) 'ref'
    exit $serving
}

# The reference's planted control is its own step: the float64 reference takes 226 s on the N1X, twice is over 10 min.
# Alone, any non-zero exit (a bad path, no GPU) would pass it, so it also needs the plant's own differing tokens.
if ($Step -eq 'control') {
    $env:INKLING2_REF_PLANT = '1'
    Timed @('inkling2-mini-ref', $Model, '32', $text) 'control' -FailOk
    Write-Output "planted control EXIT=$script:last (must be non-zero)"
    $planted = Select-String -Path "$out\control.err" -Pattern '\(PLANTED: no messages\): [1-9]\d* of 32 greedy tokens differ'
    if (-not $script:last -or -not $planted) { exit 1 }
    exit 0
}

# serve: one server at --parallel 8, greedy 256-token replies at 1, 2, 4 and 8 streams, timed by the server's own `done`
# lines (first token, decode): run 2's client timed itself, as PowerShell's line reads fell 4x behind the server's
# output. Reply bodies are kept per level so concurrent replies can be compared token for token with the solo run
# (output agreement); a prefill probe sends one short and one ~4k-token prompt for prefill tok/s and TTFT.
Pool 'before the server'
$port = 8188
$proc = Start-Process -FilePath $server -PassThru -NoNewWindow -RedirectStandardOutput "$out\server.log" `
    -RedirectStandardError "$out\server.err" -ArgumentList @('serve', "`"$Model`"", '--backend', 'cuda',
    '--host', '127.0.0.1', '--port', $port, '--name', 'inkling-2-mini', '--parallel', '8', '--context', '8192',
    '--no-update-check')
try {
    while ($true) {
        try { Invoke-RestMethod "http://127.0.0.1:$port/v1/models" -TimeoutSec 5 | Out-Null; break } catch {}
        if ($proc.HasExited -or $clock.Elapsed.TotalSeconds -gt 240) { throw 'server did not answer /v1/models' }
        Start-Sleep 1
    }
    Write-Output ("RESULT n1x server ready in {0:N1} s" -f $clock.Elapsed.TotalSeconds)
    Pool 'server loaded'
    # nvidia-smi samples the pool every 500 ms in its own process for the whole run; its largest sample is the peak
    $smi = Start-Process -FilePath nvidia-smi -PassThru -NoNewWindow -RedirectStandardOutput "$out\pool.csv" `
        -ArgumentList @('-i', $gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', '500')
    Add-Type -AssemblyName System.Net.Http
    # a request gets the step's remaining budget (run 6 used under 40 s of it), not a free 4 minutes
    $http = New-Object System.Net.Http.HttpClient
    $body = '{"model":"inkling-2-mini","prompt":"Write a short Python function that computes the Fibonacci ' +
        'sequence and explain it.","max_tokens":256,"temperature":0,"ignore_eos":true}'
    # Prefill probe, before the decode levels: one short prompt, then a ~4,000-token one repeated so its prompt pass
    # is not the template's alone. Each is one request; the done line carries prompt=, cached= and ttft= (seconds to
    # first token), so prefill tok/s is (prompt - cached) / ttft and TTFT is the line's own number, not the client's.
    $word = 'The quick brown fox jumps over the lazy dog near the river bank while engineers measure prefill throughput. '
    $long = ($word * 400)
    foreach ($probe in @(@('prefill-short', 'Explain one tick of a clock.'), @('prefill-long', $long))) {
        $tag, $prompt = $probe
        $pbody = (@{model='inkling-2-mini'; prompt=$prompt; max_tokens=8; temperature=0; ignore_eos=$true} | ConvertTo-Json -Compress)
        $http.Timeout = [TimeSpan]::FromMilliseconds((Left))
        $before = @(Select-String -Path "$out\server.log" -Pattern '\] done ').Count
        $r = $http.PostAsync("http://127.0.0.1:$port/v1/completions",
            (New-Object System.Net.Http.StringContent $pbody, ([Text.Encoding]::UTF8), 'application/json'))
        if (-not $r.Result.IsSuccessStatusCode) { throw "$tag`: $($r.Result.StatusCode)" }
        $d = @(Select-String -Path "$out\server.log" -Pattern '\] done ' | Select-Object -Skip $before)[0]
        if ($d.Line -match 'prompt=(\d+) cached=(\d+) .*ttft=([\d.]+)') {
            $rate = ([double]$Matches[1] - [double]$Matches[2]) / [double]$Matches[3]
            Write-Output ("RESULT n1x {0}: {1} prompt tokens ({2} cached), prefill {3:N0} tok/s, first token {4} s" -f $tag, $Matches[1], $Matches[2], $rate, $Matches[3])
        } else { Write-Output "RESULT n1x $tag`: done line unreadable: $($d.Line)" }
    }
    $replies = @{}
    foreach ($n in 1, 2, 4, 8) {
        $http.Timeout = [TimeSpan]::FromMilliseconds((Left))
        $before = @(Select-String -Path "$out\server.log" -Pattern '\] done ').Count
        $calls = foreach ($i in 1..$n) {
            $http.PostAsync("http://127.0.0.1:$port/v1/completions",
                (New-Object System.Net.Http.StringContent $body, ([Text.Encoding]::UTF8), 'application/json'))
        }
        foreach ($c in $calls) { if (-not $c.Result.IsSuccessStatusCode) { throw "$n streams: $($c.Result.StatusCode)" } }
        $replies[$n] = @(foreach ($c in $calls) { ($c.Result.Content.ReadAsStringAsync().Result | ConvertFrom-Json).choices[0].text })
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
    $agree = @{}
    foreach ($n in 2, 4, 8) { $agree[$n] = @($replies[$n] | Where-Object { $_ -ceq $replies[1][0] }).Count }
    Write-Output ("RESULT n1x greedy output agreement vs the solo reply: 2 streams {0}/2, 4 streams {1}/4, 8 streams {2}/8 identical" -f $agree[2], $agree[4], $agree[8])
    foreach ($n in 2, 4, 8) { if ($agree[$n] -ne $n) { throw "$n streams: $($agree[$n])/$n greedy replies agree with solo" } }
    $smi.Kill(); $smi.WaitForExit()
    $peak = (Get-Content "$out\pool.csv" | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum
    if (-not $peak) { throw "no samples in $out\pool.csv" }
    Write-Output "RESULT n1x peak GPU memory sampled every 500 ms: $peak MiB"
    Write-Output "RESULT n1x server process peak working set $([int]($proc.PeakWorkingSet64 / 1MB)) MiB"
} finally {
    foreach ($p in $proc, $smi) { if ($p -and -not $p.HasExited) { $p.Kill() } }
    Get-Content "$out\server.err" -Tail 20
}
