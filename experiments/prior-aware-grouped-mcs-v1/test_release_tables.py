import csv,gzip,json
from release_tables import copy_light

def test_large_json_exact_record_reassembly(tmp_path):
    rows=[dict(i=i,text='保留全部字段'*10) for i in range(30)]
    p=tmp_path/'source.json';p.write_text(json.dumps(rows,ensure_ascii=False))
    outputs=copy_light(p,tmp_path/'published/data.json',limit=1000)
    joined=[]
    for f in outputs[:-1]:joined+=json.loads(f.read_text());assert f.stat().st_size<=1000
    assert joined==rows

def test_gzip_csv_released_as_plain_lossless_parts(tmp_path):
    rows=[['i','text'],*[ [str(i),'comma, newline\n你好'+str(i)] for i in range(20)]]
    p=tmp_path/'source.csv.gz'
    with gzip.open(p,'wt',encoding='utf-8',newline='') as f:csv.writer(f).writerows(rows)
    outputs=copy_light(p,tmp_path/'published/table.csv',limit=160)
    joined=[]
    for part in outputs[:-1]:
        with part.open(newline='',encoding='utf-8') as f:reader=csv.reader(f);assert next(reader)==rows[0];joined+=list(reader)
        assert part.suffix=='.csv' and part.stat().st_size<=160
    assert joined==rows[1:]
