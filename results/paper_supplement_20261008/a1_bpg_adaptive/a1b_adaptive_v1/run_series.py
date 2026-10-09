import pathlib,json,subprocess,time
b=pathlib.Path(__file__).parent;q=json.loads((b/'series_request.json').read_text());rows=[]
for stage in q['stages']:
 assert time.time()<q['deadline_unix']
 argv=[q['python'],'-B',q['runner'],'--request',q['request'],'--stage',stage]
 with (b/(stage+'.console.log')).open('xb') as f:
  p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  (b/'series_progress.json').write_text(json.dumps({'stage':stage,'pid':p.pid,'prior':rows},indent=2));rc=p.wait()
 rows.append({'stage':stage,'exit_code':rc});(b/'series_progress.json').write_text(json.dumps({'prior':rows},indent=2))
 if rc:raise SystemExit(rc)
(b/'series_completion.json').write_text(json.dumps({'all_children_waited':True,'stages':rows},indent=2))
