param(
    [string]$Version = $env:AGENTOP_VERSION,
    [string]$InstallDir = "$HOME\.local\bin"
)

$ErrorActionPreference = "Stop"
$Repo = "blackboxdelta/agentop"

if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne "X64") {
    throw "agentop: this installer supports Windows x64 only."
}

if (-not $Version) {
    $release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
    $Version = $release.tag_name
}

$baseUrl = "https://github.com/$Repo/releases/download/$Version"
$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("agentop-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tempDir | Out-Null

try {
    $archive = Join-Path $tempDir "agentop.zip"
    $checksum = Join-Path $tempDir "agentop.sha256"
    Invoke-WebRequest "$baseUrl/agentop-windows-x64.zip" -OutFile $archive
    Invoke-WebRequest "$baseUrl/agentop-windows-x64.sha256" -OutFile $checksum

    $expected = ((Get-Content $checksum -Raw).Trim() -split "\s+")[0].ToLower()
    $actual = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected) {
        throw "agentop: checksum verification failed."
    }

    $package = Join-Path $tempDir "package"
    Expand-Archive $archive -DestinationPath $package
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item (Join-Path $package "agentop.exe") (Join-Path $InstallDir "agentop.exe") -Force

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $pathParts = @($userPath -split ";" | Where-Object { $_ })
    if ($InstallDir -notin $pathParts) {
        [Environment]::SetEnvironmentVariable(
            "Path",
            (($pathParts + $InstallDir) -join ";"),
            "User"
        )
        $env:Path = "$env:Path;$InstallDir"
    }

    Write-Host "agentop $Version installed to $InstallDir\agentop.exe"
    Write-Host "Open a new terminal and run: agentop"
}
finally {
    Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}
