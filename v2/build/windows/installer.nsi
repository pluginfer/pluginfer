; Pluginfer Windows installer (NSIS script template).
;
; Build:
;   makensis -DVERSION=1.0.0 v2/build/windows/installer.nsi
;
; Variables substituted by build_windows.py:
;   VERSION       semver
;   GIT_SHA       short hash
;   APP_DIR_REL   path (relative to this script) to the PyInstaller
;                  --onedir output (typically ../../dist/pluginfer/)

!define APPNAME "Pluginfer"
!define COMPANYNAME "Pluginfer Network"
!define DESCRIPTION "Distributed GPU Compute Node"
!ifndef VERSION
  !define VERSION "1.0.0"
!endif
!ifndef GIT_SHA
  !define GIT_SHA "unknown"
!endif
!ifndef APP_DIR_REL
  !define APP_DIR_REL "..\..\dist\pluginfer"
!endif

Name "${APPNAME} ${VERSION}"
OutFile "..\..\dist\Pluginfer-${VERSION}-Setup.exe"
InstallDir "$PROGRAMFILES64\Pluginfer"
RequestExecutionLevel admin
SetCompressor /SOLID lzma

; Publisher info shown in Add/Remove Programs.
VIProductVersion "1.0.0.0"
VIAddVersionKey "ProductName" "${APPNAME}"
VIAddVersionKey "CompanyName" "${COMPANYNAME}"
VIAddVersionKey "FileDescription" "${DESCRIPTION}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "InternalName" "Pluginfer"
VIAddVersionKey "OriginalFilename" "Pluginfer-${VERSION}-Setup.exe"

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "Pluginfer" SecCore
    SectionIn RO     ; required
    SetOutPath "$INSTDIR"
    File /r "${APP_DIR_REL}\*"
    WriteUninstaller "$INSTDIR\uninstall.exe"

    ; Add to Add/Remove Programs
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
        "DisplayName" "${APPNAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
        "DisplayVersion" "${VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
        "Publisher" "${COMPANYNAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
        "URLInfoAbout" "https://pluginfer.network"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
        "UninstallString" '"$INSTDIR\uninstall.exe"'
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
        "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
        "NoRepair" 1

    ; Register a Windows service via NSSM if present (deferred -- service
    ; install is the operator's choice, with a clear instruction at the
    ; finish page so the user opts in).
    DetailPrint "Pluginfer ${VERSION} installed (build ${GIT_SHA})."
    DetailPrint "To run as a service:"
    DetailPrint "  nssm install Pluginfer "$INSTDIR\pluginfer.exe""
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\uninstall.exe"
    RMDir /r "$INSTDIR"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
    ; Wallet PEM lives in %APPDATA%\Pluginfer; intentionally NOT removed so
    ; reinstall doesn't burn keys. Operator can rm -rf manually if intended.
SectionEnd
