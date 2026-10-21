"""CPU-only delivery of the closed 13-dB examples. No model imports."""
import sys,json,csv,hashlib,shutil
from pathlib import Path
import execute as x
def main():
    import numpy as np
    from PIL import Image
    reg,r=x.bind(sys.argv[1]);D=x.O/'delivery';D.mkdir()
    P=x.R/'paper/figures/selected_examples_13db_20261021_v1';P.mkdir()
    Q=P/'individual';Q.mkdir();receipts=x.O/'execution_receipts';receipts.mkdir()
    records=[]
    # Capture only closed, small controls/receipts, never the runtime cache or arrays.
    for pattern in ['*/launch.json','*/actual_wait.json','*/failure.json','*/*/launch.json','*/*/actual_wait.json','*/*/failure.json','*/completion.json']:
        for p in x.T.glob(pattern):
            v=x.read(p)
            if p.name=='launch.json':
                ident=v.get('process');now=x.process(ident['pid']) if ident else None
                assert not now or now['start_ticks']!=ident['start_ticks'],p
            target=receipts/p.relative_to(x.T);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==p.read_bytes()
            else:shutil.copyfile(p,target)
    control=x.T/'environment_recovery_control_v1'
    for stage in ['NEST_phy','ARPC_phy','BPG_phy','P100000_tx','P100000_rx','Swin200000_tx','Swin200000_rx','ALL_assemble']:
        wait=x.read(control/stage/'actual_wait.json');assert type(wait['returncode']) is int and wait['returncode']==0 and wait['child_gone']
    assert x.read(x.T/'metrics_import_recovery_dispatch_v1/actual_wait.json')['returncode']==0
    c=x.read(x.T/'metrics_import_recovery_v1/completion.json');assert len(c['rows'])==20 and c['new_pairs']==8
    arr=[];delta=[];allfiles=[]
    for ordinal,idx in enumerate(x.INDICES,1):
        s=next(v for v in reg['sources'] if v['source_index']==idx)
        source=next(v for v in r['sources'] if v['source_index']==idx)
        with np.load(x.checked(source['archive']),allow_pickle=False) as z:pixels=z['pixels'].copy()
        orig=Q/f'{ordinal:02d}_original.png';Image.fromarray(pixels.transpose(1,2,0)).save(orig)
        assert np.array_equal(np.asarray(Image.open(orig)),np.asarray(Image.open(s['original_png'])))
        for m in x.METHODS:
            row=next(v for v in c['rows'] if v['method']==m and v['source_index']==idx)
            with np.load(x.checked(row['image']),allow_pickle=False) as z:image=z['image'].copy()
            assert x.w.array_sha(image)==row['image_sha256']
            file=Q/f'{ordinal:02d}_{m}.png';Image.fromarray(np.rint(image.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(file)
            prev=x.read(x.checked(next(v for v in reg['cases'] if v['method']==m and v['source_index']==idx)['previous_metrics']))
            same=prev['image_sha256']==row['image_sha256']
            row=dict(row,N=1024,image_path=file.relative_to(x.R).as_posix(),image_png_sha256=x.sha(file),native_float_path=row['image']['path'],native_float_file_sha256=row['image']['sha256'],version='Frozen five-method timing '+x.OLD,policy_snr_db=10 if m in ('NEST','ARPC','BPG') else 13,same_native_pixels_as_10db=same)
            row.pop('image');arr.append(row)
            delta.append(dict(method=m,source_index=idx,psnr_10=prev['psnr_db'],psnr_13=row['psnr_db'],lpips_10=prev['lpips_alex'],lpips_13=row['lpips_alex'],same_pixels=same))
    for name,rows in [('display_manifest.csv',arr),('comparison_10_vs_13.csv',delta)]:
        with (D/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Actual decoder counts come from the closed PHY receipts, not an assumed logical count.
    counts={}
    for m in ['NEST','ARPC','BPG']:
        role='phy_environment_recovery_v1' if m=='NEST' else 'phy'
        complete=x.read(x.T/m/role/'completion.json');counts[m]=complete['model_calls']
    result=dict(status='COMPLETE',registration=x.pin(x.O/'registration.json'),source_indices=x.INDICES,
        N=1024,snr_db=13,logical_channel_events=20,digital_channel_events=12,continuous_channel_events=8,
        digital_source_payload_reuse=12,new_caption=0,new_tokenizer=0,new_source_probability=0,
        continuous_encoder_calls=8,new_receiver_outputs=8,reused_receiver_outputs=12,
        new_psnr_lpips_pairs=8,reused_psnr_lpips_pairs=12,all_five_methods_current=True,
        digital_actual_input_equality_checks=12,actual_digital_decoder_calls=counts,
        rejected_or_unfit=sum(v['status'].endswith('GRAY') for v in arr),
        initial_failures=[dict(stage='PHY initialization',science_calls=0,reason='visual interpreter lacked sionna'),dict(stage='metric module import',model_calls=0,pair_calls=0,reason='missing original import search path')],
        successful_scientific_calls_repeated=0,bootstrap=0,training=0,old13db_raw_HiFi_reused=False)
    x.save(D/'completion.json',result)
    for p in sorted(Q.glob('*.png')):allfiles.append(x.pin(p))
    x.save(D/'image_manifest.json',dict(files=allfiles))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
