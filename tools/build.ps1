<#
.SYNOPSIS
Configures and builds Murmur from any terminal (Cursor, VS Code, Windows Terminal),
and optionally runs one of its programs.

.EXAMPLE
.\build.cmd                                      # x64-release, every target
.\build.cmd -Run game                            # build the game and play it
.\build.cmd x64-debug -Target murmur -Test       # the game in Debug, then the unit tests
.\build.cmd -Target murmur,murmur-lab            # several targets
.\build.cmd -Fresh                               # reconfigure from scratch
.\build.cmd --help                               # the options and every -Run program
#>
param(
    [ValidateSet('x64-release', 'x64-debug')]
    [string]$Preset = 'x64-release',
    [string[]]$Target,
    [switch]$Test,
    [switch]$Fresh,
    [string]$Run,
    [switch]$Help
)

# What each -Run name builds and starts, in the order --help lists them.
$programs = [ordered]@{
    game        = @{ Target = 'murmur';       Args = @();
                     About = 'the night market' }
    demo        = @{ Target = 'murmur';       Args = @('--demo', "What's good tonight?", '--warmup', '15', '--xray');
                     About = 'a hands-free run: 15 s of chatter, then one question to Mei' }
    chat        = @{ Target = 'murmur-lab';   Args = @('chat');        About = 'talk to Mei in the terminal' }
    schedule    = @{ Target = 'murmur-lab';   Args = @('schedule');    About = 'continuous batching, step by step' }
    batch       = @{ Target = 'murmur-lab';   Args = @('batch');       About = 'throughput against latency, 1 to 64 conversations' }
    market      = @{ Target = 'murmur-lab';   Args = @('market');      About = '30 NPCs talking non-stop, then a leak check' }
    calibrate   = @{ Target = 'murmur-lab';   Args = @('calibrate');   About = 'measure channel importance for INT4 weights' }
    quality     = @{ Target = 'murmur-lab';   Args = @('quality');     About = 'what each weight format loses in quality' }
    actions     = @{ Target = 'murmur-lab';   Args = @('actions');     About = 'NPC actions with and without the grammar' }
    speculative = @{ Target = 'murmur-lab';   Args = @('speculative'); About = 'the draft model guessing ahead' }
    compare     = @{ Target = 'murmur-lab';   Args = @('compare', '--weights', 'fp8'); About = 'the shared comparison workload' }
    tests       = @{ Target = $null;          Args = @();              About = 'every unit test (the same as -Test)' }
}

function Show-Usage {
    Write-Host @'
usage: build [x64-release|x64-debug] [-Target name[,name...]] [-Test] [-Fresh] [-Run program]

  x64-release|x64-debug  the preset to build (default x64-release)
  -Target name,...       build only these targets (default: everything, or what -Run needs)
  -Test                  then run the unit tests: murmur-host-tests, and murmur-tests with CUDA
  -Fresh                 reconfigure from scratch first
  -Run program           then start one of these, from the repository root:
'@
    foreach ($name in $programs.Keys) {
        Write-Host ('    {0,-12} {1}' -f $name, $programs[$name].About)
    }
}

if ($Help) {
    Show-Usage
    exit 0
}
if ($Run -and -not $programs.Contains($Run)) {
    [Console]::Error.WriteLine("build: unknown program '$Run'`n")
    Show-Usage
    exit 1
}
if ($Run -eq 'tests') {
    $Test = $true
    $Run = ''
}

. (Join-Path $PSScriptRoot 'vs-env.ps1')

Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    # Configure only when needed: a build re-runs it by itself after CMakeLists.txt changes.
    # A configure run without the MSVC environment caches "ninja not found" for good, so
    # such a cache is thrown away.
    $cache = "out/build/$Preset/CMakeCache.txt"
    $poisoned = (Test-Path $cache) -and
        (Select-String -Path $cache -Pattern 'CMAKE_MAKE_PROGRAM:FILEPATH=.*NOTFOUND' -Quiet)
    if ($Fresh -or $poisoned -or -not (Test-Path $cache)) {
        $configure = @('--preset', $Preset)
        if ($Fresh -or $poisoned) { $configure += '--fresh' }
        cmake @configure
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }

    # The host tests always exist; the GPU tests only when CMake found CUDA.
    $tests = @('murmur-host-tests')
    if (Select-String -Path $cache -Pattern '^CMAKE_CUDA_COMPILER:\w+=(?!.*NOTFOUND).+' -Quiet) {
        $tests += 'murmur-tests'
    }

    # Asked to run something without naming targets: build only what that needs.
    # Several targets come through build.cmd as one comma-separated word.
    $targets = @($Target | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
    if ($Run -and $targets.Count -eq 0) { $targets = @($programs[$Run].Target) }
    if ($targets.Count -gt 0) {
        if ($Test) { $targets += $tests }
        if ($Run) { $targets += $programs[$Run].Target }
        $targets = $targets | Select-Object -Unique
    }
    $build = @('--build', '--preset', $Preset)
    if ($targets.Count -gt 0) { $build += '--target'; $build += $targets }
    cmake @build
    if ($LASTEXITCODE) { exit $LASTEXITCODE }

    # From the repository root, so the programs find models/ and reference/.
    if ($Test) {
        foreach ($testProgram in $tests) {
            & "out/build/$Preset/$testProgram.exe"
            if ($LASTEXITCODE) { exit $LASTEXITCODE }
        }
    }
    if ($Run) {
        $program = $programs[$Run]
        $runArgs = $program.Args
        & "out/build/$Preset/$($program.Target).exe" @runArgs
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
