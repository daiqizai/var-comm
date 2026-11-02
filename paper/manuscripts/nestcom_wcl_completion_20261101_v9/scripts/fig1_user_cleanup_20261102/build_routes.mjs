import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
const require=createRequire('C:/Users/11946/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/package.json');
const {Presentation,PresentationFile,FileBlob}=require('@oai/artifact-tool');
const H=path.dirname(fileURLToPath(import.meta.url));
const source=await PresentationFile.importPptx(await FileBlob.load(path.join(H,'user_edited_source.pptx')));
const inspection=await source.inspect({kind:'slide,shape,textbox',maxChars:4000});
await fs.writeFile(path.join(H,'artifact_import_inspection.json'),JSON.stringify(inspection,null,2));
const plan=JSON.parse(await fs.readFile(path.join(H,'route_plan.json'),'utf8'));
const p=Presentation.create({slideSize:{width:plan.slide_size_px[0],height:plan.slide_size_px[1]}}),s=p.slides.add();
for(const r of plan.routes){
 const xs=r.points.map(p=>p[0]),ys=r.points.map(p=>p[1]);
 const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(.01,Math.max(...xs)-x),h=Math.max(.01,Math.max(...ys)-y);
 s.shapes.add({name:r.name,geometry:'custom',position:{left:x*4/3,top:y*4/3,width:w*4/3,height:h*4/3},fill:'none',line:{fill:r.color,width:r.width*4/3,style:r.dash?'dash':'solid'},customPaths:[{width:w*4/3,height:h*4/3,commands:r.points.map((q,i)=>({[i?'lineTo':'moveTo']:{x:(q[0]-x)*4/3,y:(q[1]-y)*4/3}}))}]});
}
await(await PresentationFile.exportPptx(p)).save(path.join(H,'arrows_native.pptx'));
console.log(JSON.stringify({native_arrow_shapes:plan.routes.length,imported_user_slide_count:source.slides.items.length}));
