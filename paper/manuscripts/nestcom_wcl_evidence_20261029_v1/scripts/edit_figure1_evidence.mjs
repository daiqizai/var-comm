// Targeted native-PPTX edits: preserve every grid, bird, connector and panel.
import fs from'node:fs/promises';import path from'node:path';import{createRequire}from'node:module';import{fileURLToPath}from'node:url';
const require=createRequire(process.env.CODEX_NODE_MODULES+'/package.json');
const{FileBlob,PresentationFile}=require('@oai/artifact-tool');
const base=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const p=await PresentationFile.importPptx(await FileBlob.load(base+'/inputs/figure1_previous_native.pptx'));
const s=p.slides.items[0];const layout=JSON.parse(await(await s.export({format:'layout'})).text());
await fs.writeFile(base+'/validation/figure1_before_layout.json',JSON.stringify(layout,null,2));
const edits=[];
for(const e of layout.elements){
 if(e.kind!=='textbox')continue;
 if(e.text.includes('77.70')||e.text.includes('95.21')){p.delete(e.id);edits.push({id:e.id,old:e.text,action:'remove measured percentage'});}
 if(e.text.includes('1024')&&e.name.startsWith('B__')){
  const o=p.resolve(e.id);o.text='Schematic bars';o.text.style={...e.style,italic:false,typeface:'Nimbus Sans'};edits.push({id:e.id,old:e.text,new:'Schematic bars'});
 }
 if(e.text==='Update and\nadvance'){
  const o=p.resolve(e.id);o.text='Residual accumulation\nFiner-scale conditioning';o.position={left:515,top:266,width:148,height:34};o.text.style={...e.style,fontSize:10.67};edits.push({id:e.id,old:e.text,new:'Residual accumulation / Finer-scale conditioning'});
 }
 if(e.text==='Finer scales'){p.resolve(e.id).position={...e.position,top:211};}
}
await fs.mkdir(base+'/validation/pptx_build',{recursive:true});
await fs.writeFile(base+'/validation/figure1_edits.json',JSON.stringify({source:'inputs/figure1_previous_native.pptx',edits,images_changed:false,same_scale_reprediction_loop_added:false},null,2));
await fs.writeFile(base+'/validation/figure1_after_layout.json',await(await s.export({format:'layout'})).text());
await fs.writeFile(base+'/validation/pptx_build/figure1_preview.png',new Uint8Array(await(await p.export({slide:s,format:'png',scale:3})).arrayBuffer()));
await(await PresentationFile.exportPptx(p)).save(base+'/validation/pptx_build/figure1_candidate.pptx');
console.log('Native Figure 1 edited; candidate requires finalization and PowerPoint visual export.');
