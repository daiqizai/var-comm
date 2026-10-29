import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const base=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const runtime=process.env.CODEX_ARTIFACT_RUNTIME||'C:/Users/11946/.cache/codex-runtimes/codex-primary-runtime';
const skill=process.env.PRESENTATIONS_SKILL||runtime+'/plugins/openai-primary-runtime/plugins/presentations/skills/presentations';
const require=createRequire(runtime+'/dependencies/node/node_modules/package.json');
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const source=base+'/../nestcom_wcl_focused_20261029_v1/figures/figure1.pptx';
const {execFileSync}=await import('node:child_process');
const py=runtime+'/dependencies/python/python.exe';
const dim=execFileSync(py,['-c',"import zipfile,xml.etree.ElementTree as E,sys;z=zipfile.ZipFile(sys.argv[1]);e=E.fromstring(z.read('ppt/presentation.xml')).find('{http://schemas.openxmlformats.org/presentationml/2006/main}sldSz');print(e.attrib['cx']+','+e.attrib['cy'])",source],{encoding:'utf8'}).trim();
const result=await finalizePresentation({
 workspaceDir:base,candidatePath:base+'/validation/figures/figure1_candidate.pptx',
 finalPath:base+'/validation/figures/finalized/figure1_release3.pptx',pythonExecutable:py,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu',dim,'--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['Nimbus Sans','STIXGeneral'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},
 verifyArtifactToolImport:true,receiptPath:base+'/validation/figures/figure1_release3.validation.json'
});
console.log(JSON.stringify(result));
