// Local revision of the existing native diagram. Never changes image pixels or experiments.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const SCRIPT_DIR=path.dirname(fileURLToPath(import.meta.url));
const H=path.resolve(SCRIPT_DIR,'..');
const modules=process.env.RUNTIME_NODE_MODULES??process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES;
const require=modules?createRequire(path.join(modules,'package.json')):createRequire(import.meta.url);
const {FileBlob,PresentationFile}=require('@oai/artifact-tool');
const JSZip=require('jszip');
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(H,'inputs/figure1_before_recurrence_20261102.pptx')));
const s=p.slides.items[0];const L=JSON.parse(await(await s.export({format:'layout'})).text());
if(!process.argv[2])throw new Error('Provide a new private output directory');
const out=path.resolve(process.argv[2]);await fs.mkdir(out,{recursive:true});
const C={ink:'#283641',blue:'#235F8C',line:'#5488B3',orange:'#966123',purple:'#80679C',muted:'#4E5B65',fill:'#EEF3F7'};
const find=k=>L.elements.find(e=>e.name?.includes(k));
const names=k=>L.elements.filter(e=>e.name?.includes(k));
const deleted=new Set();
const remove=e=>{if(e&&!deleted.has(e.id)){p.delete(e.id);deleted.add(e.id);}};
const move=(e,x,y,w,h)=>{p.resolve(e.id).position={left:x,top:y,width:w,height:h};};
function txt(name,text,x,y,w,h,size=10,color=C.ink,bold=false){
 const o=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 o.text=text;o.text.insets=0;o.text.style={fontSize:size,typeface:'Calibri',color,bold,alignment:'center',verticalAlignment:'middle',wrap:false};return o;
}
function rect(name,x,y,w,h,fill=C.fill,stroke='#8BA4B8'){
 return s.shapes.add({name,geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:.55}});
}
function line(name,x1,y1,x2,y2,color=C.ink,arrow=true,dash=false,width=.7){
 return s.shapes.add({name,geometry:'connector',kind:'straight',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1),horizontalFlip:x2<x1,verticalFlip:y2<y1},line:{fill:color,width,style:dash?'dashed':'solid'},tail:arrow?{type:'triangle',width:'sm',length:'sm'}:{type:'none'}});
}
function elbow(name,pts,color=C.ink,dash=false,width=.7){for(let i=1;i<pts.length;i++)line(name+'_'+i,...pts[i-1],...pts[i],color,i===pts.length-1,dash,width);}
function replace(e,text,x,y,w,h,size=10,color=C.ink){const o=p.resolve(e.id);o.text=text;o.text.insets=0;o.text.style={fontSize:size,typeface:'Calibri',color,alignment:'center',verticalAlignment:'middle',wrap:false};if(x!==undefined)o.position={left:x,top:y,width:w,height:h};return o;}
function sum(name,x,y){s.shapes.add({name,geometry:'ellipse',position:{left:x,top:y,width:11,height:11},fill:'white',line:{fill:'#41596C',width:.65}});txt(name+'_plus','+',x,y-.6,11,12,11);}
function transformGroup(key,x,y,scale=1){const es=names(key);const left=Math.min(...es.map(e=>e.position.left)),top=Math.min(...es.map(e=>e.position.top));for(const e of es){const b=e.position;move(e,x+(b.left-left)*scale,y+(b.top-top)*scale,b.width*scale,b.height*scale);}}

// Native Office Math is injected into these editable textbox placeholders after export.
const math=[];
const esc=t=>String(t).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
let mathStyle={size:10.4,color:C.ink};
const rp=()=>`<a:rPr lang="en-US" sz="${Math.round(mathStyle.size*75)}"><a:solidFill><a:srgbClr val="${mathStyle.color.slice(1)}"/></a:solidFill><a:latin typeface="Cambria Math"/><a:ea typeface="Cambria Math"/><a:cs typeface="Cambria Math"/></a:rPr>`;
const mr=(t,plain=false)=>`<m:r><m:rPr><m:sty m:val="${plain?'p':'i'}"/></m:rPr>${rp()}<m:t xml:space="preserve">${esc(t)}</m:t></m:r>`;
const sub=(base,n)=>`<m:sSub><m:sSubPr/><m:e>${base}</m:e><m:sub>${Array.isArray(n)?n.map(a=>mr(a[0],a[1])).join(''):mr(n,false)}</m:sub></m:sSub>`;
const acc=(base,chr)=>`<m:acc><m:accPr><m:chr m:val="${chr}"/><m:ctrlPr>${rp()}</m:ctrlPr></m:accPr><m:e>${base}</m:e></m:acc>`;
const hat=(v,n=null)=>n===null?acc(mr(v),'̂'):sub(acc(mr(v),'̂'),n);
const tilde=(v,n)=>sub(acc(mr(v),'̃'),n);
const sym=(v,n)=>sub(mr(v),n);
const idx=n=>String(n).split(/([0-9:+−,]+)/).filter(Boolean).map(x=>[x,/^[0-9:+−,]+$/.test(x)]);
function equation(name,fn,x,y,w,h,size=10.4,color=C.ink){mathStyle={size,color};const xml=fn();const o=txt(name,'math',x,y,w,h,size,color);math.push({name,xml,size,color,x,y,w,h});return o;}
function oldEquation(key,fn,size=10.4,color=null){const e=find(key);if(!e||deleted.has(e.id))return;const b=e.position;const col=color??e.style?.color??C.ink;remove(e);equation('M_'+key,fn,b.left,b.top,b.width,b.height,size,col);}

// Use the fonts actually present in the Windows export environment, not unavailable aliases.
for(const e of L.elements){if(e.kind==='textbox'){const o=p.resolve(e.id);o.text.style={...e.style,typeface:e.style?.typeface==='STIXGeneral'?'Cambria Math':'Calibri'};}}

// Panel (a): body processing remains on the main path; paid profile joins it BEFORE AWGN.
replace(find('A__0262'),'Prefix fit +\nAC encoder',215,88,57,26,10.2,C.blue);
replace(find('A__0265'),'Body: LDPC\n+ QAM',286.5,88.6,45,24,8.5);
replace(find('A__0274'),'Body: Demap\n+ LDPC',349,88.6,45,24,8.5);
for(const e of L.elements){const m=e.name?.match(/^A__(\d+)__/);if(m&&Number(m[1])>=275&&Number(m[1])<=292)remove(e);}
for(const k of ['V2_noisy_header','V2_decoded_MCS','V2_decoded_endpoint'])names(k).forEach(remove);
rect('R_profile_header',285,127,54,24,'#FAF8FC',C.purple);
txt('R_profile_header_text','Protected\nprofile header',286,128,52,22,9.1,C.purple);
elbow('R_selected_profile_fields',[[236,114.6],[236,139],[284.5,139]],C.purple);
equation('R_selected_e',()=>mr('Selected ',true)+mr('e'),239,117,44,10,9.3,C.purple);
equation('R_fixed_a',()=>mr('fixed ',true)+mr('a'),239,127,44,10,9.3,C.purple);
elbow('R_header_to_mux',[[318,127],[318,122],[333.3,122],[333.3,101.18]],C.purple);
s.shapes.add({name:'R_mux_before_channel',geometry:'ellipse',position:{left:332.3,top:100.18,width:2,height:2},fill:C.purple,line:{fill:C.purple,width:.3}});
equation('R_total_symbols',()=>mr('68',true)+mr(' + (',true)+mr('N')+mr(' − 68) symbols',true),285,153,109,11,9.3,C.muted);
elbow('R_noisy_header',[[346.3,101.18],[346.3,125],[403,125],[403,145],[405,145]],C.purple);
txt('R_noisy_header_label','Noisy header',349,130,49,12,9.1,C.purple);
replace(find('A__0303'),'Profile\ndecode',408,134,55,23,10.2,C.purple);
elbow('R_decoded_a',[[418,132],[418,119],[386,119],[386,114.6]],C.purple,true);
line('R_decoded_e',436,132,436,114.6,C.purple,true,true);
move(find('A__0306'),390,117.2,11,12);move(find('A__0307'),440,120,11,12);
replace(find('A__0251'),'see (b)',216,150,34,11,9.5,C.blue);

// Panel (b): explicitly separate termination, actual-length decision, and the selected output.
move(find('G0014_arithmetic_box___B__1104'),84.7,194,99,36);
replace(find('B__1105'),'Independent termination\nactual-length check',86.2,195,96,22,9.2,C.blue);
equation('R_selected_prefix_endpoint',()=>mr('Selected ',true)+sym('π','e')+mr(', ',true)+mr('e'),87,218,94,11,9.8,C.blue);
replace(find('B__0985'),'Same MCS a',187,193,84,14,9.6);
replace(find('B__1108'),'Schematic bars',187,211,84,13,9.6,C.muted);
remove(find('V2_AC_check_to_capacity'));line('R_fitting_to_capacity',129,230,129,237);
remove(find('B__1141'));equation('R_capacity',()=>mr('Body bits ',true)+sym('k','a'),186,226,85,11,9.6,C.muted);

// Panel (c): preserve native history/prediction/overwrite grids and the boundary sum.
const retained=['G0015_history','G0017_grid','G0018_grid'];
names('V2_received_K').forEach(remove);
for(const e of L.elements){const b=e.position;if(b.left>=288&&b.top>=194&&b.top<307&&!retained.some(k=>e.name?.includes(k)))remove(e);}
transformGroup('G0015_history',297,221,1);
transformGroup('G0017_grid',388,218,1);
transformGroup('G0018_grid',446,218,1);
rect('R_boundary_VAR_shadow',322.8,217,53,27);rect('R_boundary_VAR',321.5,218.3,53,27);
txt('R_boundary_VAR_text','Shared VAR',323,223,50,16,10.2,'#41596C');
equation('R_coarse_tokens',()=>hat('q',idx('1:m')),291,242,33,14,10.4,C.blue);
line('R_coarse_to_VAR',312,231,321,231);line('R_VAR_to_boundary',375,231,387,231);
txt('R_argmax','argmax',385,204,37,12,9.6,C.muted);
txt('R_predict','Predict',386,245,35,11,9.6,C.orange);
line('R_prediction_to_merge',420,231,445,231);
txt('R_overwrite','Overwrite',433,245,52,11,9.6,C.blue);
for(let j=0;j<9;j++)rect('R_received_token_'+j,446+j*3.46,207,3.46,5,'#8DBBDB','#4B8DBB');
txt('R_received_K','Received K tokens',421,194,82,12,9.8,C.blue);
line('R_received_overwrite',461.5,212,461.5,217,C.blue);
equation('R_boundary_merged_label',()=>tilde('q',idx('m+1')),490,244,31,13,10.4,C.blue);
line('R_merged_to_C',478,231,490,231);
rect('R_boundary_C',491,220,28,23);
equation('R_boundary_C_math',()=>sym('C',idx('m+1')),491,221,28,21,10.4,'#41596C');
line('R_boundary_C_to_sum',520,231,532,231);sum('R_boundary_sum',533,225.5);
equation('R_coarse_latent',()=>hat('f','m'),526,203,25,16,10.4,'#41596C');line('R_coarse_to_sum',538.5,218,538.5,225);
line('R_sum_to_initial_state',545,231,553,231);
equation('R_initial_state',()=>hat('f',idx('m+1')),552,222,42,19,11,C.blue);
txt('R_residual_label','Residual accumulation',548,199,127,14,10.3,'#41596C');

// Updated state has TWO explicit consumers: next-scale context and the residual addition.
elbow('R_initial_state_to_loop',[[572,241],[572,258],[390.5,258],[390.5,267]],C.blue);
line('R_retained_state_to_addition',572,258,572,267,C.blue);
txt('R_context_branch_label','Finer-scale\nconditioning',292,264,61,22,9.7,'#41596C');
rect('R_next_context',363,267,55,23);
txt('R_next_context_text','Context /\nresize',365,268,51,21,9.6,'#41596C');
line('R_context_to_VAR',419,278.5,428,278.5,C.blue);
rect('R_finer_VAR',429,267,43,23);
txt('R_finer_VAR_text','Shared\nVAR',430,268,41,21,9.6,'#41596C');
line('R_finer_VAR_to_update',473,278.5,507,278.5,C.orange);
equation('R_finer_tokens',()=>tilde('q','s'),478,264,25,13,10.4,C.orange);
rect('R_recurrence_update',509,267,112,23);
equation('R_recurrence_equation',()=>hat('f','s')+mr(' = ',true)+hat('f',idx('s−1'))+mr(' + ',true)+sym('C','s')+mr('(',true)+tilde('q','s')+mr(')',true),511,269,108,19,10.2,'#41596C');
// Feedback is routed wholly inside (c); it feeds the next context and retained-state input.
elbow('R_feedback_next_scale',[[621,278.5],[627,278.5],[627,301],[354,301],[354,258],[390.5,258]],C.blue);
line('R_state_feedback_rejoin',390.5,258,572,258,C.blue,false);
equation('R_scale_range',()=>mr('s')+mr(' = ',true)+mr('m')+mr('+2, …, 10',true),398,291,100,10,9.6,C.orange);
txt('R_repeat_label','Next scale',536,291,71,10,9.3,C.blue);
line('R_final_to_decoder',628,278.5,647,278.5,C.blue);
equation('R_final_latent',()=>hat('f',idx('10')),624,261,26,13,10.4,C.blue);
equation('R_final_Dc',()=>sym('D','c'),649,267,26,22,11,'#41596C');

// Replace all remaining math-like labels, including old glyph-only pseudo-subscripts.
for(const [key,fn] of [
 ['A__0011',()=>mr('x')],['A__0014',()=>mr('E')],['A__0026',()=>mr('f')],['A__0029',()=>sym('R','s')],['A__0037',()=>sym('C','s')],['A__0045',()=>sym('R',idx('s+1'))],
 ['A__0248',()=>mr('1 … ',true)+mr('m')],['A__0249',()=>mr('m')+mr('+1',true)],['A__0250',()=>mr('e')+mr(' = (',true)+mr('m')+mr(',',true)+mr('K')+mr(')',true)],
 ['A__0257',()=>sym('q',idx('<s'))],['A__0270',()=>mr('n')],['A__0306',()=>mr('a')],['A__0307',()=>mr('e')],
 ['A__0922',()=>hat('q',idx('<s'))],['A__0924',()=>mr('1 … ',true)+mr('m')],['A__0925',()=>mr('m')+mr('+1',true)],['A__0926',()=>mr('… 10',true)],
 ['A__0957',()=>sym('C','s')],['A__0960',()=>mr('∑',true)],['A__0971',()=>hat('f',idx('10'))],['A__0974',()=>sym('D','c')],['A__0979',()=>hat('x')],
 ['B__0986',()=>mr('Prefix ',true)+sym('π','e')],['B__1098',()=>mr('1 … ',true)+mr('m')],['B__1099',()=>mr('m')+mr('+1',true)],
 ['B__1148',()=>mr('b')+mr('(',true)+mr('x')+mr(',',true)+mr('e')+mr(') + 13 + 16 ≤ ',true)+sym('k','a')]
])oldEquation(key,fn);

// Uniform body typefaces; the original picture and grid shapes remain native and intact.
await fs.writeFile(path.join(out,'after_layout.json'),await(await s.export({format:'layout'})).text());
await fs.writeFile(path.join(out,'math_manifest.json'),JSON.stringify(math,null,2));
const native=path.join(out,'artifact_native.pptx');await(await PresentationFile.exportPptx(p)).save(native);
const z=await JSZip.loadAsync(await fs.readFile(native));let xml=await z.file('ppt/slides/slide1.xml').async('string');
const replacements=new Map(math.map(e=>[e.name,e]));let count=0;
xml=xml.replace(/<p:sp\b[\s\S]*?<\/p:sp>/g,shape=>{
 const name=shape.match(/<p:cNvPr[^>]*\bname="([^"]*)"/)?.[1];const f=replacements.get(name);if(!f)return shape;
 const sz=Math.round(f.size*75),color=f.color.slice(1);
 const body=`<p:txBody xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="ctr"><a:noAutofit/></a:bodyPr><a:lstStyle/><a:p><a:pPr algn="ctr"><a:defRPr sz="${sz}"><a:solidFill><a:srgbClr val="${color}"/></a:solidFill><a:latin typeface="Cambria Math"/></a:defRPr></a:pPr><a14:m xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMathPara><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr><m:oMath>${f.xml}</m:oMath></m:oMathPara></a14:m><a:endParaRPr lang="en-US" sz="${sz}"/></a:p></p:txBody>`;
 const native=shape.replace(/<p:txBody>[\s\S]*?<\/p:txBody>/,body);
 count++;return `<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"><mc:Choice xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main" Requires="a14">${native}</mc:Choice><mc:Fallback>${shape}</mc:Fallback></mc:AlternateContent>`;
});
if(count!==math.length)throw new Error(`Native math replacement count ${count}/${math.length}`);
// Artifact Tool does not currently serialize the wrap:false text option.
// Keep explicit line breaks, with native PowerPoint no-wrap text bodies.
xml=xml.replace(/<a:bodyPr\b[^>]*>/g,tag=>tag.replace(/\s+wrap="[^"]*"/g,'').replace(/(\/?>)$/,' wrap="none"$1'));
z.file('ppt/slides/slide1.xml',xml);
const dest=path.join(out,'figure1_math_native.pptx');await fs.writeFile(dest,await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log(JSON.stringify({destination:dest,native_math_objects:count,panels:3,original_canvas_emu:'6547104,3108960',font_policy:['Calibri','Cambria Math']}));
