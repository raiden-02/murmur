# Loads the MSVC x64 toolchain (cl, link, ninja, the Windows SDK) into the current
# PowerShell, so the CMake presets work in any terminal, not only inside Visual Studio.
# Dot-source it:  . .\tools\vs-env.ps1   (build.cmd and Cursor's terminal do this for you)

if ($env:VSCMD_ARG_TGT_ARCH -eq 'x64' -and (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    return
}

$installer = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer'
$vswhere = Join-Path $installer 'vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw 'Visual Studio not found. Install Visual Studio 2022 or newer with "Desktop development with C++".'
}
$vs = & $vswhere -latest -products * -version '[17.0,' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
    throw 'No Visual Studio 2022 or newer with the C++ x64 tools ("Desktop development with C++") found.'
}

# The developer shell's own scripts call vswhere by name, but it isn't on PATH by default.
$env:PATH += ";$installer"
Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
Write-Host "MSVC x64 environment loaded from $vs"
