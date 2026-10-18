; Inno Setup script for the Windows release installer.
;
; Built by .github/workflows/release.yml, which calls ISCC.exe with:
;   Version      the release version, e.g. 0.1.0
;   ExePath      the built murmuration-gpu.exe
;   ReadmePath   README.md
;   LicensePath  LICENSE
;   OutputDir    the directory the setup .exe is written to
;
; There is no icon in the repository, so the installer and the shortcuts take
; Inno's default. The release still ships a .zip for anyone who would rather
; not run an installer.

[Setup]
AppName=Murmuration GPU
AppVersion={#Version}
AppVerName=Murmuration GPU {#Version}
AppPublisher=Imanol Gomez
AppPublisherURL=https://github.com/ImanolGo/murmuration-gpu
AppSupportURL=https://github.com/ImanolGo/murmuration-gpu/issues
DefaultDirName={autopf}\Murmuration GPU
DefaultGroupName=Murmuration GPU
DisableProgramGroupPage=yes
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=murmuration-gpu-{#Version}-x86_64-windows-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile={#LicensePath}
UninstallDisplayName=Murmuration GPU {#Version}
UninstallDisplayIcon={app}\murmuration-gpu.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#ExePath}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#ReadmePath}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#LicensePath}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Murmuration GPU"; Filename: "{app}\murmuration-gpu.exe"
Name: "{autodesktop}\Murmuration GPU"; Filename: "{app}\murmuration-gpu.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\murmuration-gpu.exe"; Description: "{cm:LaunchProgram,Murmuration GPU}"; Flags: nowait postinstall skipifsilent
