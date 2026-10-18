"""Post-exit audit: only here may source truth be compared with fixed RX outputs."""
import csv,hashlib,json,pathlib,sys,time
import numpy as np

def write_csv(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    cfg=json.loads(pathlib.Path(sys.argv[1]).read_text());run=pathlib.Path(cfg['run']);O=pathlib.Path(cfg['output'])
    for name in ['tx','rx']:
        wait=json.loads((run/(name+'.actual_wait.json')).read_text())
        assert type(wait['returncode']) is int and wait['returncode']==0 and not wait['timed_out']
        pid=wait['process']['pid'];pp=pathlib.Path(f'/proc/{pid}/stat')
        if pp.exists():assert int(pp.read_text().split(') ',1)[1].split()[19])!=wait['process']['start_ticks']
    tx=json.loads((run/'tx/completion.json').read_text());rx=json.loads((run/'rx/completion.json').read_text())
    assert tx['status']==rx['status']=='COMPLETE' and rx['blocked_reads']==[]
    assert tx['counts']=={'model_load':2,'caption':8,'tokenizer':8,'tx_probability':8,'reference_render':24}
    assert rx['counts']=={'model_load':1,'rx_prefix':24,'rx_render':24}
    tref={x['key']:x for x in tx['outputs']};rows=[];lengths=[]
    for r in rx['outputs']:
        key=r['key'];sid=key.split('_')[0];e=r['endpoint'];reference=tref[key]
        truth=json.loads((run/'tx'/(sid+'_source.json')).read_text())
        assert r['caption_bytes_hex']==truth['caption_bytes_hex']
        assert r['cdf_hashes']==truth['cdf_hashes'][:e],('CDF_MISMATCH',key)
        with np.load(run/'tx'/(sid+'_truth.npz'),allow_pickle=False) as a,np.load(run/'rx'/(key+'_decoded.npz'),allow_pickle=False) as b:
            assert len(b.files)==e
            for i in range(1,e+1):assert np.array_equal(a[f's{i}'],b[f's{i}']),('TOKEN_MISMATCH',key,i)
        with np.load(reference['path'],allow_pickle=False) as a,np.load(r['path'],allow_pickle=False) as b:
            maximum=float(np.max(np.abs(a['float_output']-b['float_output'])))
            assert maximum<=1e-6,('PIXEL_TOLERANCE_FAILED',key,maximum)
            assert np.array_equal(a['pixels'],b['pixels']),('QUANTIZED_PIXELS_DIFFER',key)
        rows.append({'key':key,'source_id':truth['source_id'],'endpoint':e,'caption_exact':True,'tokens_exact':True,
            'CDFs_exact':True,'quantized_pixels_exact':True,'float_max_abs_error':maximum,'pixel_std':r['pixel_std'],
            'status':'PASS','rx_pixel_sha256':r['pixel_sha256']})
    assert len(rows)==24
    for p in sorted((run/'tx').glob('*_source.json')):
        d=json.loads(p.read_text())
        for x in d['lengths']:lengths.append({'source_id':d['source_id'],'source_index':d['source_index'],**x})
    assert len(lengths)==56
    for s in lengths:
        s['native_visual_plus_help_bpp']=s['original_visual_plus_help_bits']/(256*256)
        s['standalone_payload_bpp']=s['payload_bits']/(256*256)
        s['future_outer_length_bits']=13;s['future_outer_crc_bits']=16
    write_csv(O/'codec_roundtrip.csv',rows);write_csv(O/'source_bits_breakdown.csv',lengths)
    fits=[{'source_id':s['source_id'],'endpoint':s['endpoint'],'payload_bits':s['payload_bits'],'capacity_bits':c,
      'fits':s['payload_bits']<=c,'standalone_includes_paid_text_and_boundaries':True} for s in lengths for c in [1245,2520,4751]]
    write_csv(O/'fit_by_capacity.csv',fits)
    counts={'caption_calls':8,'tokenizer_calls':8,'source_probability_traversals':8,'prefix_roundtrips':24,
        'reference_image_generations':24,'independent_receiver_generations':24,'wireless_frames':0,'training_updates':0}
    report={'status':'ROUNDTRIPS_PASS_RESOURCE_AND_VISUAL_REVIEW_PENDING','counts':counts,'audit_time':time.time(),'native_resolution':[256,256],
        'float_tolerance_predeclared':1e-6,'all_24_exact_quantized_pixels':True,'new_channel_frames':0,
        'low_variance_outputs':[r['key'] for r in rows if r['pixel_std']<=1.0],
        'bounds_not_performance_claim':'No wireless effect is inferred from clean reconstruction.'}
    with (O/'a1_closure.json').open('x') as f:json.dump(report,f,indent=2)
    # A display-only contact sheet; saved scientific arrays are never altered.
    from PIL import Image,ImageDraw
    sources=json.loads((O/'sources.json').read_text())['a1_first8']
    size=128;gap=8;cell_h=152
    sheet=Image.new('RGB',(7*(size+gap)+gap,8*cell_h+32),'white');draw=ImageDraw.Draw(sheet)
    headings=['Source','Native e1','RX e1','Native e4','RX e4','Native e7','RX e7']
    for j,h in enumerate(headings):draw.text((gap+j*(size+gap),4),h,fill='black')
    for i,s in enumerate(sources):
        sid=f"{s['rank']:02d}";entries=[(pathlib.Path(s['archive']['path']),'source')]
        for e in [1,4,7]:entries.extend([(run/'tx'/f'{sid}_e{e}.npz','reference'),(run/'rx'/f'{sid}_e{e}.npz','receiver')])
        for j,(p,role) in enumerate(entries):
            with np.load(p,allow_pickle=False) as z:pixels=z['pixels'].copy()
            im=Image.fromarray(pixels.transpose(1,2,0)).resize((size,size),Image.Resampling.LANCZOS)
            x=gap+j*(size+gap);y=28+i*cell_h;sheet.paste(im,(x,y));draw.text((x,y+size+2),f"rank {s['rank']} / idx {s['source_index']}",fill='black')
    sheet.save(run/'a1_visual_contact.png',dpi=(150,150))
    closed=[]
    for root in [run/'tx',run/'rx',run/'wire_inputs']:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or 'runtime_cache' in p.parts:continue
            assert not p.is_symlink();a=p.stat();h=hashlib.sha256(p.read_bytes()).hexdigest();b=p.stat()
            assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
            closed.append({'path':str(p),'bytes':a.st_size,'sha256':h})
    with (O/'a1_closed_files.json').open('x') as f:json.dump({'files':closed,'contact_preview':str(run/'a1_visual_contact.png'),
        'note':'Contact image is display-only, 128 px tiles. Scientific output remains full native 256 arrays.'},f,indent=2)
    print(json.dumps(report))
if __name__=='__main__':main()
