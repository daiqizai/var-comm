"""Bind existing displayed pixels and native-array scores; no inference."""
from pathlib import Path
import csv,json,hashlib,shutil
P=Path(__file__).resolve().parents[1];W=P.parents[2]
old=list(csv.DictReader((P/'inputs/existing_figure3_metrics.csv').open()))
ada=list(csv.DictReader((W/'results/wcl_adatoken_20261027_v1/delivery/figure3_ada_metrics.csv').open()))
dest=P/'figures/individual';dest.mkdir(exist_ok=True);rows=[]
for index,subject,row in((95,'Pastries','03'),(21,'Dog','01')):
 original=W/f'paper/figures/selected_examples_10db_20261020_v1/individual/{row}_original.png'
 shutil.copyfile(original,dest/f'{index:04d}_original.png')
 for method in('NEST','ADA_PREDCLASS_FIXEDFRAME','P100000','ARPC'):
  candidates=[r for r in (ada if method.startswith('ADA')else old)if r['method']==method and int(r['source_index'])==index];assert len(candidates)==1
  r=candidates[0];assert r['snr_db']=='10'
  image=r['display_png']if method.startswith('ADA')else r['image_path'];h=r['display_png_sha256']if method.startswith('ADA')else r['image_png_sha256']
  b=(W/image).read_bytes();assert hashlib.sha256(b).hexdigest()==h
  local=f'figures/individual/{index:04d}_{method}.png';(P/local).write_bytes(b)
  rows.append(dict(source_index=index,source_id=r['source_id'],subject=subject,method=method,N=1024,snr_db=10,noise_seed=r['noise_seed'],noise_namespace=r['noise_namespace'],timing_case=r.get('timing_case','independent development2 event'),receive_status=r.get('receive_status'),lpips_alex=r['lpips_alex'],dreamsim=r['dreamsim'],image_path=local,image_png_sha256=h,input_repository_path=image,native_image_path=r['image_path']if method.startswith('ADA')else r['float_image_path'],native_file_sha256=r['image_file_sha256']if method.startswith('ADA')else r['float_image_file_sha256']))
with(P/'figure3_metrics.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Bound 8 reconstructed images, 2 sources, existing native scores.')
