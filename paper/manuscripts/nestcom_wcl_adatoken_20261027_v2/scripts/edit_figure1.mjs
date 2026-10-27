// Focused edits of the user's native-object PPTX. No flattened picture replacement.
import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
const require=createRequire(process.env.CODEX_NODE_MODULES+'/package.json');
const {FileBlob,PresentationFile,Presentation}=require('@oai/artifact-tool');
const [source,output,privateDir]=process.argv.slice(2);
if(!source||!output||!privateDir)throw Error('source PPTX, output folder, private folder required');
await fs.mkdir(output,{recursive:true});await fs.mkdir(privateDir,{recursive:true});
let p=await PresentationFile.importPptx(await FileBlob.load(source));
const slide=p.slides.items[0];const layout=JSON.parse(await(await slide.export({format:'layout'})).text());
const remove=new Set(['sh/kzydsfal','sh/wrmdkzel','sh/sn21crid','sh/upkje10j']);
function yy(y){return y<190?y*.92:y<339?174.8+(y-190)*.85:306+(y-339);}
for(const e of layout.elements){
 if(remove.has(e.id)){p.delete(e.id);continue;}
 const obj=p.resolve(e.id);const b=e.position;
 if(!b)continue;
 const height=e.kind==='textbox'?b.height:yy(b.top+b.height)-yy(b.top);
 obj.position={...b,top:yy(b.top),height};
}
function text(id,value,position){const o=p.resolve(id);o.text=value;if(position)o.position={...o.position,...position};}
text('sh/fqtk76h4','Update and\nadvance',{left:525,top:263,width:81,height:29});
text('sh/hsv29wza','Overwrite',{top:266});
text('sh/svuhcn6l','Predict',{top:266});
text('sh/tob25wzy','Finer scales',{top:266});
text('sh/wnax436d','Length + CRC',{left:101,width:67});
text('sh/alsf2to7','b(x,e) + 13 + 16 ≤ kₐ',{left:91,top:290,width:164,height:15});
// Leave native symbol/subscript labels and the measured utilization bars intact.
const proto=p.toProto();await fs.writeFile(privateDir+'/edited_proto_keys.json',JSON.stringify({keys:Object.keys(proto),slide:Object.keys(proto.slides?.[0]||{})},null,2));
// The imported page is 7.16 inches wide. Cropping unused bottom whitespace only.
proto.slides[0].heightEmu=Math.round(326.4*9525);
p=Presentation.load(proto);
const s=p.slides.items[0];
await fs.writeFile(privateDir+'/figure1.layout.json',await(await s.export({format:'layout'})).text());
await fs.writeFile(output+'/figure1.png',new Uint8Array(await(await p.export({slide:s,format:'png',scale:3})).arrayBuffer()));
await(await PresentationFile.exportPptx(p)).save(privateDir+'/figure1_candidate.pptx');
console.log('Figure 1 source edits exported; validation pending.');
