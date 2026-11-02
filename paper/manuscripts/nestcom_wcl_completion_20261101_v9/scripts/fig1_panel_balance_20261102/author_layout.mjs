import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
const require=createRequire('C:/Users/11946/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/package.json');
const {Presentation,PresentationFile,FileBlob}=require('@oai/artifact-tool');
const H=path.dirname(fileURLToPath(import.meta.url));
const plan=JSON.parse(await fs.readFile(path.join(H,'layout_plan.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(H,'source_clean.pptx')));
await fs.writeFile(path.join(H,'artifact_before.json'),JSON.stringify(await p.inspect({kind:'slide,shape,textbox,layout',maxChars:10000}),null,2));
const shapes=p.slides.items[0].shapes.items;
const byName=new Map(shapes.map(s=>[s.name,s]));
let count=0;const unresolved=[];
for(const [name,v] of Object.entries(plan.updates)){
 const s=byName.get(name);if(!s){unresolved.push(name);continue;}
 const [x,y,w,h]=v.after;s.position={left:x*4/3,top:y*4/3,width:w*4/3,height:h*4/3};count++;
}
await(await PresentationFile.exportPptx(p)).save(path.join(H,'artifact_layout.pptx'));
await fs.writeFile(path.join(H,'artifact_layout_receipt.json'),JSON.stringify({resolved_native_shapes:count,unresolved_group_containers:unresolved},null,2));
// Author the updated routes as native vector objects. Their compact heads are
// retained as the second custom path when merged into the original package.
const ap=Presentation.create({slideSize:{width:515.5*4/3,height:278.375*4/3}}),as=ap.slides.add();
for(const r of plan.newroutes){
 const xs=r.points.map(p=>p[0]),ys=r.points.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(.01,Math.max(...xs)-x),h=Math.max(.01,Math.max(...ys)-y);
 as.shapes.add({name:r.name,geometry:'custom',position:{left:x*4/3,top:y*4/3,width:w*4/3,height:h*4/3},fill:'none',line:{fill:r.color,width:r.width*4/3,style:r.dash?'dash':'solid'},customPaths:[{width:w*4/3,height:h*4/3,commands:r.points.map((q,i)=>({[i?'lineTo':'moveTo']:{x:(q[0]-x)*4/3,y:(q[1]-y)*4/3}}))}]});
}
await(await PresentationFile.exportPptx(ap)).save(path.join(H,'arrows_layout.pptx'));
console.log(JSON.stringify({edited_native_shapes:count,unresolved:unresolved.length,new_route_objects:plan.newroutes.length}));
