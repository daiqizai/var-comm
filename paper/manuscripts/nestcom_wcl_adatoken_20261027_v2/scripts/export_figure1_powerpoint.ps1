$ErrorActionPreference='Stop'
$base=Join-Path (Split-Path $PSScriptRoot -Parent) 'figures'
$app=$null;$deck=$null
try {
 $app=New-Object -ComObject PowerPoint.Application
 $deck=$app.Presentations.Open((Join-Path $base 'figure1.pptx'),$true,$false,$false)
 $deck.SaveAs((Join-Path $base 'figure1.pdf'),32)
} finally {
 if($null -ne $deck){$deck.Close();[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($deck)}
 if($null -ne $app){$app.Quit();[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)}
}
& 'C:/Users/11946/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe' -singlefile -png -r 150 (Join-Path $base 'figure1.pdf') (Join-Path $base 'figure1_print_size')
