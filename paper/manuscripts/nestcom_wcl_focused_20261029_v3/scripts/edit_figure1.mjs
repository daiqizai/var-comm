// Edit the source PPTX's native objects. Preserves its birds, residual VQ and scale grids.
import fs from 'node:fs/promises';import path from 'node:path';import {createRequire} from 'node:module';import {fileURLToPath} from 'node:url';
const require=createRequire(process.env.CODEX_NODE_MODULES+'/package.json');const {FileBlob,PresentationFile}=require('@oai/artifact-tool');
const P=path.dirname(path.dirname(fileURLToPath(import.meta.url)));const source=path.join(P,'../nestcom_wcl_focused_20261029_v1/figures/figure1.pptx');
const out=path.join(P,'validation/figures');await fs.mkdir(out,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(source));const s=p.slides.items[0];const L=JSON.parse(await(await s.export({format:'layout'})).text());
await fs.writeFile(path.join(out,'figure1_before_layout.json'),JSON.stringify(L,null,2));
const find=(t)=>L.elements.find(e=>e.name?.includes(t));const names=(t)=>L.elements.filter(e=>e.name?.includes(t));
const remove=(e)=>{if(e)p.delete(e.id)};const move=(e,x,y,w,h)=>{p.resolve(e.id).position={left:x,top:y,width:w,height:h}};
function txt(name,text,x,y,w,h,size=10.4,color='#283641',font='Nimbus Sans',bold=false){let o=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});o.text=text;o.text.insets=0;o.text.style={fontSize:size,typeface:font,color,bold,alignment:'center',verticalAlignment:'middle',wrap:false};return o}
function rect(name,x,y,w,h,fill='#F1F5F8',stroke='#8BA4B8'){return s.shapes.add({name,geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:.55}})}
function line(name,x1,y1,x2,y2,color='#283641',arrow=true,dash=false){const o=s.shapes.add({name,geometry:'connector',kind:'straight',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1),horizontalFlip:x2<x1,verticalFlip:y2<y1},line:{fill:color,width:.7,style:dash?'dashed':'solid'},tail:arrow?{type:'triangle',width:'sm',length:'sm'}:{type:'none'}});return o;}
function elbow(name,pts,color='#283641',dash=false){for(let i=1;i<pts.length;i++)line(name+'_'+i,...pts[i-1],...pts[i],color,i===pts.length-1,dash)}
function replace(e,text,x,y,w,h,size=10.4,color=null){const o=p.resolve(e.id);o.text=text;o.text.insets=0;o.text.style={...e.style,fontSize:size,...(color?{color}:{}),verticalAlignment:'middle',wrap:false};if(x!==undefined)o.position={left:x,top:y,width:w,height:h};return o}
function groupTransform(key,x,y,scale=1){const es=names(key),left=Math.min(...es.map(e=>e.position.left)),top=Math.min(...es.map(e=>e.position.top));for(const e of es){const b=e.position;move(e,x+(b.left-left)*scale,y+(b.top-top)*scale,b.width*scale,b.height*scale);}}
// (a): eliminate the former profile line that bypassed the noisy link.
for(const n of ['A__0289','A__0290','A__0304','A__0305'])remove(find(n));
replace(find('A__0303'),'Profile\ndecode',408,134,55,24,10);
replace(find('A__0274'),'Demap\n+ LDPC',349,89,45,25,10.4);
replace(find('A__0275'),'Fixed frame\nN symbols',288,115,56,21,9.6).text.style={fontSize:9.6,typeface:'Nimbus Sans',italic:false,color:'#283641',alignment:'center'};
replace(find('A__0291'),'Header: 68\n(protected)',285,145,61,20,9.3);
replace(find('A__0292'),'Body: N − 68',341,148,57,13,9.4);
replace(find('A__0306'),'a',402,120,12,13,10.4);
replace(find('A__0307'),'e',440,117,12,14,10.4);
replace(find('A__0926'),'… 10',544,132,27,16,10.4);
elbow('V2_noisy_header',[[346,102],[346,129],[403,129],[403,145],[405,145]],'#80679C');
txt('V2_noisy_header_label','Noisy header',350,119,62,10,9.4,'#80679C');
elbow('V2_decoded_MCS',[[418,132],[418,118],[386,118],[386,114]],'#80679C',true);
line('V2_decoded_endpoint',436,132,436,114,'#80679C',true,true);

// (b): distinguish independent candidate coding, capacity bits and all three endpoint cases.
replace(find('B__0984'),'Within-scale endpoints and frame fitting',32,176,243,17,10.5,'#235F8C');
replace(find('B__0985'),'Same MCS a',178,193,87,14,9.6);
replace(find('B__1108'),'Schematic bars',178,207,87,14,9.6);
remove(find('B__1109'));
move(find('G0014_arithmetic_box___B__1104'),84.7,196,89,27);
replace(find('B__1105'),'Independent AC +\nactual-length check',88,197,82,24,9.3,'#235F8C');
remove(find('B__1106'));names('B__1107').forEach(remove);
line('V2_AC_check_to_capacity',129,223,129,237);
replace(find('B__1141'),'Body information bits kₐ',137,223,134,11,9.4,'#4E5B65');
replace(find('B__1114'),'Whole (m,0)',12,232.5,71,14.9,9.6);
replace(find('B__1129'),'Next (m+1,0)',9,247.2,75,14.9,9.6);
replace(find('B__1121'),'Within (m,K)',9,261.8,75,14.9,9.6,'#235F8C');
for(const e of L.elements){const m=e.name?.match(/^B__(\d+)__/);if(m&&Number(m[1])>=1110&&Number(m[1])<=1140){const o=p.resolve(e.id);o.position={...o.position,top:o.position.top+4};}}
// (c): retain the existing native grids/history while replacing the ambiguous serial update block.
const kept=['G0015_history','G0017_grid','G0018_grid','G0021_grid'];
for(const e of L.elements){const b=e.position;if(b.left>=288&&b.top>=194&&b.top<307&&!kept.some(k=>e.name?.includes(k)))remove(e);}
groupTransform('G0015_history',297,221,1);
groupTransform('G0017_grid',388,218,.74);
groupTransform('G0018_grid',446,218,.74);
groupTransform('G0021_grid',653,272,.43);
rect('V2_VAR_boundary_shadow',322.8,217,53,27);rect('V2_VAR_boundary',321.5,218.3,53,27);
txt('V2_VAR_boundary_text','Shared VAR',323,223,50,16,10.2,'#41596C');
txt('V2_coarse_tokens','q̂₁:ₘ',289,242,39,19,11.2,'#235F8C','STIXGeneral');
line('V2_coarse_to_VAR',312,231,321,231);line('V2_VAR_to_prediction',375,231,387,231);
txt('V2_argmax','argmax',385,204,37,13,9.6,'#4E5B65');txt('V2_predict','Predict',386,247,35,15,10,'#966123');
line('V2_prediction_to_merge',420,231,445,231);txt('V2_overwrite','Overwrite',422,247,59,15,9.6,'#235F8C');
for(let j=0;j<9;j++)rect('V2_received_token_'+j,446+j*3.46,207,3.46,5,'#8DBBDB','#4B8DBB');
txt('V2_received_K','Received K tokens',422,193,82,13,9.5,'#235F8C');line('V2_received_overwrite',461.5,212,461.5,217,'#235F8C');
txt('V2_merged_q','q̃ₘ₊₁',447,263,32,16,11.1,'#235F8C','STIXGeneral');
line('V2_merged_to_C',478,231,490,231);
rect('V2_C_boundary',491,220,28,23,'#EEF3F7');txt('V2_C_boundary_text','Cₘ₊₁',491,222,28,19,11,'#41596C','STIXGeneral');
line('V2_C_to_sum',520,231,532,231);
s.shapes.add({name:'V2_boundary_sum',geometry:'ellipse',position:{left:533,top:226,width:11,height:11},fill:'white',line:{fill:'#41596C',width:.6}});txt('V2_boundary_plus','+',533,225,11,13,11);
txt('V2_prior_latent','f̂ₘ',525,201,28,18,11.2,'#41596C','STIXGeneral');line('V2_prior_to_sum',538.5,218,538.5,225);
line('V2_sum_to_state',545,231,553,231);
txt('V2_updated_state','f̂ₘ₊₁',552,221,42,20,12,'#235F8C','STIXGeneral');
txt('V2_residual_heading','Residual accumulation',548,199,130,15,10.5,'#41596C');
line('V2_residual_retention',593,231,641,231,'#235F8C');
s.shapes.add({name:'V2_final_sum',geometry:'ellipse',position:{left:642,top:225.5,width:12,height:12},fill:'white',line:{fill:'#41596C',width:.6}});txt('V2_final_plus','+',642,225,12,14,11);
line('V2_final_to_Dc',655,231,661,231);txt('V2_final_Dc','D꜀',661,221,16,20,11,'#41596C','STIXGeneral');

// The merged state drives finer prediction via resizing. No return to the boundary grid.
elbow('V2_updated_to_context',[[572,242],[572,259],[562,259],[562,271]],'#235F8C');
rect('V2_context',528,271,69,28,'#EEF3F7');txt('V2_context_text','Next-scale\ncontext / resize',530,273,65,24,9.8,'#41596C');
line('V2_context_to_VAR',598,285,605,285,'#235F8C');rect('V2_finer_VAR',605,273,42,25,'#EEF3F7');txt('V2_finer_VAR_text','Shared\nVAR',607,275,38,22,9.8,'#41596C');
line('V2_VAR_to_finer',647,285,652,285);elbow('V2_finer_contribution',[[664,272],[664,255],[648,255],[648,238]],'#966123');
txt('V2_finer_C','Cₛ',660,242,17,16,10.4,'#966123','STIXGeneral');
txt('V2_condition_heading','Finer-scale\nconditioning',393,278,106,25,10.6,'#41596C');
txt('V2_finer_scales','s = m+2,…,10',598,298,77,9,9.2,'#966123');
// Keep the footer and panel boundary stable; the very last lower label remains inside panel.
const finalLayout=await(await s.export({format:'layout'})).text();await fs.writeFile(path.join(out,'figure1_after_layout.json'),finalLayout);
await(await PresentationFile.exportPptx(p)).save(path.join(out,'figure1_candidate.pptx'));
console.log('Edited native figure candidate exported.');
