import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FileBlob,PresentationFile} from 'file:///C:/Users/11946/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const H=process.env.NEST_FIGURE_WORK_DIR;
if(!H)throw new Error('Set NEST_FIGURE_WORK_DIR to an existing scratch directory');
const P=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const p=await PresentationFile.importPptx(await FileBlob.load(P+'/inputs/system_diagram_v3_entropy.pptx'));
const snap=await p.inspect({kind:'slide,shape,textbox,image',maxChars:1000000});
const items=snap.ndjson.trim().split('\n').map(x=>JSON.parse(x));
const slide=p.slides.items[0];
// Reuse the source deck, slide and selected module objects. Reflow only this figure.
const retained=['sh/k3yl0zql','sh/7qp4be9c','sh/65g3298r','sh/54r2poju','sh/qds7e1kn','sh/9kfex8vi'];
const ids=[...new Set(items.filter(x=>['shape','textbox','image'].includes(x.kind)&&!retained.includes(x.id)).map(x=>x.id))];
p.delete(ids);
const blue='#3775BA',green='#5E966B',ink='#17212A',gray='#767676';
function box(name,x,y,w,h,t,fill='white',stroke=gray,fsz=27,bold=false){
 const s=slide.shapes.add({name,geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,
  line:{fill:stroke,width:1.5},text:{text:t,style:{fontFamily:'Arial',fontSize:fsz,color:ink,bold,alignment:'center',verticalAlignment:'middle',insets:9}}});
 return s;
}
function label(name,x,y,w,h,t,fsz=26,bold=false,align='left'){
 const s=box(name,x,y,w,h,t,'none','none',fsz,bold);s.text.style={fontFamily:'Arial',fontSize:fsz,color:ink,bold,alignment:align,verticalAlignment:'middle',insets:0};return s;
}
function reuse(id,x,y,w,h,t,fsz=28,bold=true){
 const s=p.resolve(id);s.position={left:x,top:y,width:w,height:h};s.text=t;s.fill='none';s.line={fill:'none',width:0};
 s.text.style={fontFamily:'Arial',fontSize:fsz,color:ink,bold,alignment:'left',verticalAlignment:'middle',insets:0};return s;
}
function arrow(a,b,from='right',to='left',dashed=false){
 return slide.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:ink,width:2,style:dashed?'dashed':'solid'},tail:{type:'triangle',width:'sm',length:'sm'}});
}
reuse('sh/k3yl0zql',10,5,1750,40,'(a) Fixed-budget entropy-coded transmission and recovery',31);
reuse('sh/7qp4be9c',10,53,790,35,'Transmitter',29);
reuse('sh/65g3298r',960,53,800,35,'Receiver',29);
const policy=reuse('sh/qds7e1kn',190,95,600,70,'Frozen policy (N, SNR): target endpoint and MCS\nPer-image fallback uses actual terminated length',26,false);
const shared=reuse('sh/9kfex8vi',960,95,800,70,'Shared frozen VAR prior and codebook\nRX reconstructs CDFs from received token history',26,false);
const tok=box('tokenizer',10,190,180,155,'Visual encoder\n+ multiscale\nresidual VQ','#F1F6FB',blue);
reuse('sh/54r2poju',10,350,180,68,'Frozen tokenizer\nRGB 256 × 256',26,false);
const prefix=box('prefix_length',225,190,280,155,'Within-scale prefix\nActual-length check\nLterm + 13 + 16 ≤ k','#FFF8E8','#B49448');
const tx=box('digital_transmitter',540,190,260,155,'Terminated AC\nLength / CRC / pad\nLDPC + QAM','#F1F6FB',blue);
const channel=box('AWGN',835,190,100,155,'AWGN\ny = s+n\nEs = 2','#F4F1F7','#8064A2',26);
const decode=box('digital_receiver',970,190,225,155,'Header + body\nQAM / LDPC\nCRC + AC','#F0F7F2',green);
const completion=box('prefix_reconstruction',1230,190,285,155,'Received token prefix\nPredict and overwrite\nGenerate finer scales','#F0F7F2',green);
const dc=box('latent_reconstruction',1550,190,215,155,'Codebook sum\nFrozen Dc\nRGB image','#F0F7F2',green);
arrow(tok,prefix);arrow(prefix,tx);arrow(tx,channel);arrow(channel,decode);arrow(decode,completion);arrow(completion,dc);
label('prefix_detail',225,350,280,76,'Complete scales 1…m\n+ first K of scale m+1',26,false,'center');
label('paid_header',540,350,395,84,'Header: 68 symbols\nBody: 956 symbols\nTotal: 1024 symbols',26,false,'left');
label('failure',970,350,795,70,'Rejected header / payload / source decode: fixed gray output\nDecoded tokens are recovered; untransmitted tokens are generated.',26);
label('util_heading',10,460,1750,40,'Entropy-bitstream capacity utilization · 10 dB · common500',29,true);
label('util_definition',10,505,1750,36,'Terminated entropy bits / (k − 13 length bits − 16 CRC bits); usable capacity = 2520 bits',26);
for(const [i,name,value,color]of [[0,'WHOLE',.7770373015873016,gray],[1,'PARTIAL',.9520801587301587,blue]]){
 label('util_name_'+name,10,554+51*i,210,37,name,27,true);
 box('util_track_'+name,230,555+51*i,1180,34,'','#ECEEF0','#BFC3C7',26);
 box('util_fill_'+name,230,555+51*i,1180*value,34,'',color,color,26);
 label('util_value_'+name,1440,549+51*i,310,46,(100*value).toFixed(2)+'%',28,true);
}
label('boundary_title',10,673,1750,40,'(b) Prefix-constrained reconstruction · example m = 7, K = 75',31,true);
const prior=box('decoded_context',10,745,245,130,'Decoded scales 1–7\n155 token indices','#F1F6FB',blue,27);
const predict=box('boundary_predict',295,745,320,130,'Predict all 100 tokens\nof boundary scale 8\nfrom decoded context','#FFF8E8','#B49448',27);
const overwrite=box('boundary_overwrite',655,745,350,130,'Overwrite first 75\nwith received tokens\nKeep 25 predictions','#F1F6FB',blue,27);
const advance=box('advance',1045,745,320,130,'Advance with the\nmerged scale-8 state\n(75 received + 25 predicted)','#F0F7F2',green,27);
const fine=box('finer_scales',1405,745,360,130,'Generate finer scales\nScale 9: 169 tokens\nScale 10: 256 tokens','#FFF8E8','#B49448',27);
arrow(prior,predict);arrow(predict,overwrite);arrow(overwrite,advance);arrow(advance,fine);
label('receiver_rule',10,898,1750,53,'Raster order fixes the received positions. Only received tokens overwrite predictions; source truth is unavailable at RX.',26);
slide.speakerNotes.text='Rearranged from the user-supplied system_diagram_v3_entropy.pptx. All diagram objects remain editable. Utilization uses results/n1024_consistency_20261019_v1/family_snr_summary.csv. Dimensions are normalized to 7.16 inches in the deterministic export step.';
await fs.writeFile(H+'/figure1_layout.json',await(await slide.export({format:'layout'})).text());
await(await PresentationFile.exportPptx(p)).save(H+'/figure1_draft.pptx');
console.log('draft saved');
