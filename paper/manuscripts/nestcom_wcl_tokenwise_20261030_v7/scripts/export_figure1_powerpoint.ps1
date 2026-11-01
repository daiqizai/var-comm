$ErrorActionPreference='Stop'
$base=Join-Path (Split-Path $PSScriptRoot -Parent) 'figures'
if (Get-Process -Name POWERPNT -ErrorAction SilentlyContinue) {
 throw 'Close PowerPoint before this isolated export; an existing user session will not be closed by this script.'
}
$poppler=$env:PDFTOPPM
if (-not $poppler) {
 $cmd=Get-Command pdftoppm -ErrorAction SilentlyContinue
 if ($cmd) {$poppler=$cmd.Source} else {
  $poppler='C:/Users/11946/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe'
 }
}
if (-not (Test-Path -LiteralPath $poppler)) {throw 'Set PDFTOPPM to the installed Poppler pdftoppm executable.'}
$app=$null;$deck=$null
try {
 $app=New-Object -ComObject PowerPoint.Application
 $deck=$app.Presentations.Open((Join-Path $base 'figure1.pptx'),$true,$false,$false)
 $deck.SaveAs((Join-Path $base 'figure1.pdf'),32)
} finally {
 if($null -ne $deck){$deck.Close();[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($deck)}
 if($null -ne $app){$app.Quit();[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)}
}
& $poppler -singlefile -png -r 150 (Join-Path $base 'figure1.pdf') (Join-Path $base 'figure1_print_size')
if ($LASTEXITCODE -ne 0) {throw 'Poppler figure preview export failed.'}
