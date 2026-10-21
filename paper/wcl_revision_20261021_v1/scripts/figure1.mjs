import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,PresentationFile} from 'file:///C:/Users/11946/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const P=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..').replaceAll('\\','/');
const H=process.env.NEST_FIGURE_WORK_DIR;
if(!H)throw new Error('Set NEST_FIGURE_WORK_DIR to a separate writable scratch directory');
await fs.mkdir(H,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(P+'/inputs/system_diagram_v3_entropy.pptx'));
const snap=await p.inspect({kind:'slide,shape,textbox,image',maxChars:1000000});
const items=snap.ndjson.trim().split('\n').map(x=>JSON.parse(x)),slide=p.slides.items[0];
const retained=items.filter(x=>['sh/vyp0je1s','sh/gzqlgb69'].includes(x.id)||
 (x.kind==='shape'&&x.name?.startsWith('multi_scale_tokens__')&&!x.name.includes('arrowEnd')&&x.position.top>=249&&x.position.top<361&&x.position.width<10&&x.position.height>1)||
 (x.kind==='shape'&&x.position.top>=650&&x.position.top<732&&x.position.left>=40&&x.position.left<372&&x.position.width<25&&x.position.height>1));
const keep=new Set(retained.map(x=>x.id));p.delete([...new Set(items.filter(x=>['shape','textbox','image'].includes(x.kind)&&!keep.has(x.id)).map(x=>x.id))]);
const blue='#3775BA',green='#5E966B',ink='#17212A',gray='#767676',gold='#D49B3D';
function box(name,x,y,w,h,t,fill='white',stroke=gray,sz=28,bold=false){return slide.shapes.add({name,geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:1.6},text:{text:t,style:{fontFamily:'Arial',fontSize:sz,color:ink,bold,alignment:'center',verticalAlignment:'middle',insets:8}}});}
function label(name,x,y,w,h,t,sz=27,bold=false,align='left'){const s=box(name,x,y,w,h,t,'none','none',sz,bold);s.text.style={fontFamily:'Arial',fontSize:sz,color:ink,bold,alignment:align,verticalAlignment:'middle',insets:0};return s;}
function arrow(a,b,from='right',to='left',dash=false){return slide.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:ink,width:2.2,style:dash?'dashed':'solid'},tail:{type:'triangle',width:'sm',length:'sm'}});}
// Preserve the actual source-deck thumbnails and multiscale grid artwork.
for(const it of retained){const s=p.resolve(it.id),q=it.position;
 if(it.id==='sh/vyp0je1s')s.position={left:13,top:210,width:97,height:95};
 else if(it.id==='sh/gzqlgb69')s.position={left:1660,top:210,width:98,height:95};
 else if(it.name.startsWith('multi_scale_tokens__'))s.position={left:330+(q.left-314)*.94,top:199+(q.top-250)*.94,width:q.width*.94,height:q.height*.94};
 else s.position={left:55+(q.left-41)*.88,top:566+(q.top-650)*.88,width:q.width*.88,height:q.height*.88};
}
label('panel_a',12,0,1750,40,'(a) Budget-adaptive next-scale token communication',31,true);
label('tx_heading',13,53,945,40,'Transmitter',29,true);
label('rx_heading',1130,53,630,40,'Receiver',29,true);
const policy=box('policy',443,92,507,54,'Budget + SNR → endpoint + MCS','#F4F7FA','#B6C2CC',27);
label('frozen_legend',1125,98,635,44,'Shared frozen tokenizer, VAR prior and decoder',25,false);
const input=p.resolve('sh/vyp0je1s'),output=p.resolve('sh/gzqlgb69');
const tok=box('tokenizer',145,184,160,145,'VAR\ntokenizer','#F1F6FB',blue,29,true);
const prefix=box('prefix',443,184,248,145,'Within-scale prefix\nEntropy coding\nActual-length fallback','#FFF8E8',gold,27);
const tx=box('tx',725,184,225,145,'Protected header\n+ payload\nLDPC + QAM','#F1F6FB',blue,27);
const awgn=box('channel',980,184,120,145,'AWGN\nchannel','#F4F1F7','#8064A2',27,true);
const rx=box('rx',1130,184,260,145,'Header / body decode\nEntropy decode\nReceived prefix','#F0F7F2',green,27);
const rec=box('reconstruction',1420,184,210,145,'VAR completion\nLatent reconstruction\nDc','#F0F7F2',green,27);
arrow(input,tok);arrow(prefix,tx);arrow(tx,awgn);arrow(awgn,rx);arrow(rx,rec);arrow(rec,output);
// Anchors around the retained multiscale artwork keep the flow continuous.
const ga=box('grid_anchor',328,205,86,100,'','none','none');arrow(tok,ga);arrow(ga,prefix);
arrow(policy,prefix,'bottom','top',true);arrow(policy,tx,'bottom','top',true);
label('input_caption',3,337,120,40,'Image',25,false,'center');
label('tokens_caption',307,337,127,66,'Multiscale\ntokens',25,false,'center');
label('output_caption',1635,337,137,40,'Image',25,false,'center');
label('frame_budget',443,345,658,43,'68 header + 956 payload symbols = 1024',25,false,'center');
label('reject',1130,345,500,43,'Rejected frame → fixed gray output',25,false,'center');
label('panel_b',12,413,1750,42,'(b) Prefix-constrained reconstruction',31,true);
label('context_title',30,492,350,58,'Decoded coarser scales',28,true,'center');
label('context_detail',35,665,350,55,'Scales 1–7',25,false,'center');
label('predict_title',442,492,340,58,'Predict boundary scale',28,true,'center');
label('overwrite_title',852,492,394,58,'Overwrite received positions',28,true,'center');
function grid(name,x,y,sz,received){
 const a=box(name+'_anchor',x,y,sz,sz,'','none','none');
 for(let i=0;i<100;i++){
  const isRx=i<received,xx=x+i%10*sz/10,yy=y+Math.floor(i/10)*sz/10;
  box(name+'_'+i,xx,yy,sz/10,sz/10,'',isRx?'#B9DBFB':'#FFF0CF',isRx?blue:gold,25);
  if(!isRx){const z=box(name+'_hatch_'+i,xx+2,yy+2,sz/10-4,sz/10-4,'','none','none');
   slide.shapes.add({name:name+'_diagonal_'+i,geometry:'line',position:{left:xx+1,top:yy+1,width:sz/10-2,height:sz/10-2},line:{fill:gold,width:.8}});}
 }
 return a;
}
const context=box('context_anchor',44,561,325,90,'','none','none');
const predicted=grid('predicted',532,559,151,0),merged=grid('merged',970,559,151,75);
const future=box('finer',1374,566,362,130,'Generate finer scales\nfrom the merged state','#FFF8E8',gold,29,true);
arrow(context,predicted);arrow(predicted,merged);arrow(merged,future);
label('predict_note',442,722,340,42,'Boundary scale',25,false,'center');
label('merged_note',844,722,406,42,'75 received + 25 predicted',25,false,'center');
box('received_legend',1377,722,22,22,'','#B9DBFB',blue);label('received_label',1410,713,140,42,'Received',25);
box('pred_legend',1570,722,22,22,'','#FFF0CF',gold);label('pred_label',1603,713,157,42,'Predicted',25);
const photo=await fs.readFile(P+'/inputs/diagram_thumbnail.png');
for(const xx of [14,1661])slide.images.add({blob:photo.buffer.slice(photo.byteOffset,photo.byteOffset+photo.byteLength),contentType:'image/png',alt:'Original schematic bird thumbnail',fit:'contain',position:{left:xx,top:211,width:95,height:93}});
slide.speakerNotes.text='Simplified from the user original system_diagram_v3_entropy.pptx. Original thumbnails and multiscale token grid objects retained. Capacity-utilization evidence moved to Table I. Two panels: system flow and predict–overwrite–advance mechanism. Source/rules: original released NeST implementation, m=7,K=75 illustration. No scientific data generated.';
await fs.writeFile(H+'/figure1_layout.json',await(await slide.export({format:'layout'})).text());
await fs.writeFile(H+'/figure1_retained_objects.json',JSON.stringify(retained,null,2));
await(await PresentationFile.exportPptx(p)).save(H+'/figure1_draft.pptx');
console.log('source-deck figures retained, simplified draft saved');
