param([Parameter(Mandatory=$true)][string]$InputPptx,[Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$sourceZip=[System.IO.Compression.ZipFile]::OpenRead($InputPptx)
try {
 $reader=[System.IO.StreamReader]::new($sourceZip.GetEntry('ppt/presentation.xml').Open())
 try { $sourceXml=$reader.ReadToEnd() } finally { $reader.Dispose() }
 $sourceSize=[regex]::Match($sourceXml,'<p:sldSz\b[^>]*/>').Value
 if(-not $sourceSize){throw 'No source slide dimensions'}
} finally { $sourceZip.Dispose() }
if(Get-Process -Name POWERPNT -ErrorAction SilentlyContinue){throw 'An existing PowerPoint session is open; this isolated exporter will not close user sessions.'}
[void][System.IO.Directory]::CreateDirectory($OutputDirectory)
$app=$null;$deck=$null
try{
 $app=New-Object -ComObject PowerPoint.Application
 Write-Output 'Opening native source...'
 $deck=$app.Presentations.Open($InputPptx,$true,$false,$false)
 Write-Output 'Source opened; saving editable copy...'
 $deck.SaveCopyAs((Join-Path $OutputDirectory 'figure1.pptx'),24,-1)
 Write-Output 'Saving PDF...'
 $deck.SaveAs((Join-Path $OutputDirectory 'figure1.pdf'),32)
 $result=@{slides=$deck.Slides.Count;width_points=$deck.PageSetup.SlideWidth;height_points=$deck.PageSetup.SlideHeight;native_powerpoint_version=$app.Version;embed_true_type_fonts_requested=$true;source=$InputPptx;pdf=(Join-Path $OutputDirectory 'figure1.pdf')}
 $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'native_export.json') -Encoding utf8
}finally{
 if($null-ne $deck){$deck.Close();[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($deck)}
 if($null-ne $app){$app.Quit();[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)}
}
# Preserve the exact original EMU canvas after PowerPoint's SaveCopyAs rounding.
$savedZip=[System.IO.Compression.ZipFile]::Open((Join-Path $OutputDirectory 'figure1.pptx'),[System.IO.Compression.ZipArchiveMode]::Update)
try {
 $entry=$savedZip.GetEntry('ppt/presentation.xml')
 $reader=[System.IO.StreamReader]::new($entry.Open())
 try { $xml=$reader.ReadToEnd() } finally { $reader.Dispose() }
 $xml=[regex]::Replace($xml,'<p:sldSz\b[^>]*/>',$sourceSize)
 $entry.Delete()
 $replacement=$savedZip.CreateEntry('ppt/presentation.xml')
 $writer=[System.IO.StreamWriter]::new($replacement.Open(),[System.Text.UTF8Encoding]::new($false))
 try {$writer.Write($xml)} finally {$writer.Dispose()}
} finally {$savedZip.Dispose()}
Write-Output 'Native PowerPoint export completed.'
