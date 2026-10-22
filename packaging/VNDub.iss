; VNDub installer (Inno Setup 6.3 or newer). Build: packaging\build_app.ps1 (does everything), or only the installer:
;   packaging\build_app.ps1 -InstallerOnly                 -> installer\VNDub-Setup.exe from the existing dist\VNDub
;   iscc /DAppVersion=0.6.1 packaging\VNDub.iss            -> the same, by hand
; Per-user install (no administrator rights): %LOCALAPPDATA%\Programs\VNDub, Start-menu shortcut, entry under HKCU in
; "Apps & features". The one exception is the Microsoft Visual C++ Redistributable the models need: when the runtime
; of the machine is missing or older than 14.40, Setup runs {app}\redist\vc_redist.x64.exe, which shows Microsoft's own
; UAC prompt (declining it, or any failure, never fails the setup; Diagnostics in the app offers it again). MinGit
; (Git for Windows, portable) is installed to {app}\git for the environment installer, and bgi-extractor (BGI game
; archives, built from source: upstream publishes no binaries) to {app}\bin\bgi-extractor. All of them come from
; dist\VNDub (packaging\build_app.ps1 adds them). The app keeps its data (models, backend environments, projects, backups
; of game files, logs) in its own data folder (default %LOCALAPPDATA%\VNDub, see vndub_app/appdirs.py);
; uninstalling keeps that folder unless the user picks "delete" in the question asked by the uninstaller
; (silent uninstall: unins000.exe /VERYSILENT /DELETEDATA=1).
; Encoding: UTF-8 with BOM (the custom messages are Russian; Inno Setup before 6.3 needs the BOM, 6.3+ accepts it).

#if Ver < EncodeVer(6, 3, 0)
  #error VNDub.iss needs Inno Setup 6.3 or newer (UTF8Decode, x64compatible)
#endif

; These folders are added by packaging\build_app.ps1; compiling by hand without them would give an installer on which
; no model starts (no vcruntime140.dll), no environment installs (no git) and BGI games cannot be read.
#if !FileExists(SourcePath + "..\dist\VNDub\git\cmd\git.exe")
  #error dist\VNDub\git\cmd\git.exe is missing: run packaging\build_app.ps1 (or -InstallerOnly) first
#endif
#if !FileExists(SourcePath + "..\dist\VNDub\redist\vc_redist.x64.exe")
  #error dist\VNDub\redist\vc_redist.x64.exe is missing: run packaging\build_app.ps1 (or -InstallerOnly) first
#endif
#if !FileExists(SourcePath + "..\dist\VNDub\bin\bgi-extractor\bgi-extractor.exe")
  #error dist\VNDub\bin\bgi-extractor\bgi-extractor.exe is missing: run packaging\build_app.ps1 (or -InstallerOnly) first
#endif

#define AppName "VNDub"
#ifndef AppVersion
  #define AppVersion "0.6.1"
#endif
#define AppPublisher "VNDub"
#define AppExe "VNDub.exe"

[Setup]
; AppId identifies the install across versions (upgrade in place, one uninstall entry): never change it
AppId={{6F2D8E54-9C3B-4A71-B0E5-3D7A1C95F8A2}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
; per-user only: the data folder lives in the user's profile too
PrivilegesRequired=lowest
OutputDir=..\installer
OutputBaseFilename=VNDub-Setup
SetupIconFile=VNDub.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
UsePreviousAppDir=yes
UsePreviousLanguage=yes
ShowLanguageDialog=auto
LanguageDetectionMethod=uilanguage

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
russian.DeleteDataInstruction=Удалить данные VNDub?
russian.DeleteDataText=Папка данных:%n%1%n%nВ ней модели, окружения нейросетей, проекты, резервные копии оригинальных файлов игр (они нужны, чтобы откатить патч) и журналы.
russian.DeleteDataKeep=Оставить данные%nРекомендуется. Модели не придётся скачивать заново, проекты можно будет продолжить после повторной установки.
russian.DeleteDataDelete=Удалить данные%nВсё из папки данных будет стёрто безвозвратно.
english.DeleteDataInstruction=Delete the VNDub data?
english.DeleteDataText=Data folder:%n%1%n%nIt holds the models, neural network environments, projects, backups of the original game files (needed to roll a patch back) and logs.
english.DeleteDataKeep=Keep the data%nRecommended. The models will not have to be downloaded again, and projects can be continued after a reinstall.
english.DeleteDataDelete=Delete the data%nEverything in the data folder will be erased permanently.
russian.VCRedistInstalling=Установка Microsoft Visual C++ (среда выполнения, без неё не запускаются модели)...%nWindows может запросить права администратора: подтвердите запрос.
russian.VCRedistFailed=Не удалось установить среду выполнения Microsoft Visual C++ (запрос Windows отклонён или установка не удалась).%n%nVNDub установлен. Без этой среды модели не запустятся: раздел «Диагностика» в программе предложит установить её ещё раз.
english.VCRedistInstalling=Installing Microsoft Visual C++ (runtime; the models do not start without it)...%nWindows may ask for administrator permission: please confirm the prompt.
english.VCRedistFailed=The Microsoft Visual C++ runtime could not be installed (the Windows prompt was declined or the installation failed).%n%nVNDub is installed. The models cannot start without this runtime: Diagnostics in the app offers to install it again.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; an upgrade replaces the bundled runtime wholesale (stale files of the old build must not linger)
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\git"
Type: filesandordirs; Name: "{app}\redist"
Type: filesandordirs; Name: "{app}\bin"

[Files]
; the whole dist\VNDub folder: VNDub.exe, _internal, git\ (MinGit), redist\ (vc_redist.x64.exe) and bin\
; (bgi-extractor). The redistributable stays on disk after Setup so that the app can offer to run it again.
Source: "..\dist\VNDub\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[UninstallDelete]
; the bundled runtime may hold files created at run time (caches); the data folder is never inside it.
; Git, the redistributable and the bundled tools go with the rest of {app}; the system-wide Visual C++ runtime is never
; uninstalled. (No folder in the data folder is called bin: the tools the app downloads live in <data>\tools.)
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\git"
Type: filesandordirs; Name: "{app}\redist"
Type: filesandordirs; Name: "{app}\bin"

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
const
  // the Visual C++ 2015-2022 runtime (HKLM, 64-bit view); binaries built with VS 2022 17.10+ (llama.cpp, torch 2.8) need 14.40+
  VCRuntimeKey = 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  VCMinMajor = 14;
  VCMinMinor = 40;

var
  RemoveData: Boolean;
  DataDir: String;
  DataDirFromEnv: Boolean;

function LocationFile(): String;
begin
  Result := ExpandConstant('{userappdata}\VNDub\location.json');
end;

// "data_dir" of %APPDATA%\VNDub\location.json (the app writes {"data_dir": "C:\\...\\VNDub"} as UTF-8);
// '' when the file is absent or has no such value.
function DataDirFromLocationFile(): String;
var
  Raw: AnsiString;
  S: String;
  I: Integer;
begin
  Result := '';
  if not LoadStringFromFile(LocationFile(), Raw) then
    Exit;
  S := UTF8Decode(Raw);
  I := Pos('"data_dir"', S);
  if I = 0 then
    Exit;
  I := I + Length('"data_dir"');
  // skip to the opening quote of the value
  while I <= Length(S) do
  begin
    if S[I] = '"' then
      Break;
    I := I + 1;
  end;
  I := I + 1;
  // read the JSON string: \\ -> \, \/ -> /, \" -> "
  while I <= Length(S) do
  begin
    if S[I] = '"' then
      Break;
    if (S[I] = '\') and (I < Length(S)) then
      I := I + 1;
    Result := Result + S[I];
    I := I + 1;
  end;
end;

// The same order as the app (vndub_app/appdirs.py data_dir()): VNDUB_DATA, then location.json, then the default.
function FindDataDir(): String;
begin
  Result := GetEnv('VNDUB_DATA');
  DataDirFromEnv := Result <> '';
  if Result = '' then
    Result := DataDirFromLocationFile();
  if Result = '' then
    Result := ExpandConstant('{localappdata}\VNDub');
  Result := RemoveBackslashUnlessRoot(Result);
end;

// Removes what the app writes into its data folder (keep this list in step with vndub_app/appdirs.py; a test
// checks it). The folder itself goes only if nothing else is left in it: the user may have pointed the data
// folder at a folder that holds other files, and those must survive.
procedure RemoveAppData(const Dir: String);
begin
  DelTree(Dir + '\backups', True, True, True);
  DelTree(Dir + '\configs', True, True, True);
  DelTree(Dir + '\envs', True, True, True);
  DelTree(Dir + '\logs', True, True, True);
  DelTree(Dir + '\models', True, True, True);
  DelTree(Dir + '\projects', True, True, True);
  DelTree(Dir + '\reports', True, True, True);
  DelTree(Dir + '\tools', True, True, True);
  DelTree(Dir + '\voices', True, True, True);
  DelTree(Dir + '\webview', True, True, True);
  DeleteFile(Dir + '\app.lock');
  DeleteFile(Dir + '\app.instance.json');
  DeleteFile(Dir + '\custom_models.json');
  DeleteFile(Dir + '\settings.json');
  DeleteFile(Dir + '\jobs.json');
  DeleteFile(Dir + '\throughput.json');
  DeleteFile(Dir + '\model_selftests.json');
  RemoveDir(Dir);
end;

// Installed=1 and version 14.40 or newer (the same rule as vndub_app/doctor.py decide_vcruntime).
function VCRuntimeIsCurrent(Installed, Major, Minor: Cardinal): Boolean;
begin
  Result := (Installed = 1) and ((Major > VCMinMajor) or ((Major = VCMinMajor) and (Minor >= VCMinMinor)));
end;

// True when vc_redist.x64.exe has to run: the runtime is not registered or is older than 14.40. The result is logged.
function VCRuntimeMissingOrOld(): Boolean;
var
  Installed, Major, Minor: Cardinal;
begin
  Result := True;
  if RegQueryDWordValue(HKLM64, VCRuntimeKey, 'Installed', Installed) and
     RegQueryDWordValue(HKLM64, VCRuntimeKey, 'Major', Major) and
     RegQueryDWordValue(HKLM64, VCRuntimeKey, 'Minor', Minor) then
  begin
    Log('VC++ runtime in the registry: Installed=' + IntToStr(Installed) + ' Major=' + IntToStr(Major) + ' Minor=' + IntToStr(Minor));
    Result := not VCRuntimeIsCurrent(Installed, Major, Minor);
  end
  else
    Log('VC++ runtime: no values under HKLM64\' + VCRuntimeKey);
end;

// Runs {app}\redist\vc_redist.x64.exe /install /quiet /norestart when the runtime is missing or too old. It is a
// WiX bootstrapper that elevates itself, so Windows shows its own UAC prompt although Setup runs per user. Success
// codes: 0, 1638 (a newer one is already installed), 3010 (done, restart advised). Declined prompt, failure or a
// missing file are logged and never fail the setup: the app's Diagnostics offers the redistributable again.
procedure InstallVCRuntime();
var
  Redist, Version: String;
  Code: Integer;
  Failed: Boolean;
begin
  if not VCRuntimeMissingOrOld() then
  begin
    Log('VC++ runtime check: 14.40 or newer is installed, vc_redist.x64.exe is skipped');
    Exit;
  end;
  Redist := ExpandConstant('{app}\redist\vc_redist.x64.exe');
  if not FileExists(Redist) then
  begin
    Log('VC++ runtime check: missing or older than 14.40, but ' + Redist + ' is not there: nothing to run');
    Exit;
  end;
  if not GetVersionNumbersString(Redist, Version) then
    Version := 'unknown';
  Log('VC++ runtime check: missing or older than 14.40, running ' + Redist + ' (file version ' + Version + ') /install /quiet /norestart');
  if not WizardSilent then
  begin
    WizardForm.StatusLabel.Caption := CustomMessage('VCRedistInstalling');
    WizardForm.FilenameLabel.Caption := '';
  end;
  Failed := True;
  if Exec(Redist, '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, Code) then
  begin
    if (Code = 0) or (Code = 1638) or (Code = 3010) then
    begin
      Failed := False;
      Log('vc_redist.x64.exe exit code ' + IntToStr(Code) + ': success (1638 = a newer runtime is installed, 3010 = a restart is advised)');
    end
    else
      Log('vc_redist.x64.exe exit code ' + IntToStr(Code) + ': the runtime was not installed (prompt declined or install failed); Setup continues');
  end
  else
    Log('vc_redist.x64.exe could not be started: ' + SysErrorMessage(Code) + '; Setup continues');
  if Failed then
    SuppressibleMsgBox(CustomMessage('VCRedistFailed'), mbInformation, MB_OK, IDOK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  // the files (and so the redistributable) are in place; the finish page comes next
  if CurStep = ssPostInstall then
    InstallVCRuntime();
end;

// Setup closes a running VNDub by itself (CloseApplications); Uninstall does not, and the app has no AppMutex, so
// the uninstaller closes the copy that runs from this folder: the window first, the process if it ignores that.
// Other copies of VNDub.exe (another folder, a dev build) are left alone. Needs Windows PowerShell (always there).
procedure StopRunningApp();
var
  Script, Ps: String;
  Code: Integer;
begin
  Script :=
    'param([string]$Exe)' + #13#10 +
    'Get-Process -Name VNDub -ErrorAction SilentlyContinue | ForEach-Object {' + #13#10 +
    '  $w = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $_.Id)' + #13#10 +
    '  if ($w -and $w.ExecutablePath -eq $Exe) {' + #13#10 +
    '    [void]$_.CloseMainWindow()' + #13#10 +
    '    if (-not $_.WaitForExit(15000)) { $_.Kill(); [void]$_.WaitForExit(5000) }' + #13#10 +
    '  }' + #13#10 +
    '}' + #13#10;
  Ps := ExpandConstant('{tmp}\vndub-stop.ps1');
  if SaveStringToFile(Ps, Script, False) then
  begin
    Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + Ps + '" -Exe "' + ExpandConstant('{app}\{#AppExe}') + '"',
      '', SW_HIDE, ewWaitUntilTerminated, Code);
    Log('stopped running VNDub (powershell exit code ' + IntToStr(Code) + ')');
  end;
end;

// "Keep" is the first button (Yes, the default one), "Delete" the second (No); a suppressed box means "keep".
function AskDeleteData(): Boolean;
var
  Labels: TArrayOfString;
begin
  SetArrayLength(Labels, 2);
  Labels[0] := CustomMessage('DeleteDataKeep');
  Labels[1] := CustomMessage('DeleteDataDelete');
  Result := SuppressibleTaskDialogMsgBox(CustomMessage('DeleteDataInstruction'),
    FmtMessage(CustomMessage('DeleteDataText'), [DataDir]), mbConfirmation, MB_YESNO, Labels, 0, IDYES) = IDNO;
end;

// The data folder stays unless the user asks for it to go: unins000.exe /DELETEDATA=1 (any mode), or the answer
// to the question below (shown only when the data folder exists and the uninstall is not silent).
function InitializeUninstall(): Boolean;
begin
  Result := True;
  DataDir := FindDataDir();
  RemoveData := DirExists(DataDir) and (ExpandConstant('{param:DELETEDATA|0}') = '1');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    // asked here, after Uninstall's own "are you sure?" and just before anything is removed
    if (not RemoveData) and (not UninstallSilent) and DirExists(DataDir) then
      RemoveData := AskDeleteData();
    Log('VNDub data folder: ' + DataDir + '; delete it: ' + IntToStr(Ord(RemoveData)));
    StopRunningApp();
  end;
  if (CurUninstallStep = usPostUninstall) and RemoveData then
  begin
    RemoveAppData(DataDir);
    // location.json only points at the folder just deleted (not used when VNDUB_DATA decides)
    if not DataDirFromEnv then
    begin
      DeleteFile(LocationFile());
      RemoveDir(ExtractFileDir(LocationFile()));
    end;
  end;
end;
