"""Live control identities, real waits and call ledgers; no launches or retries."""
from opt_common import *

def main():
 rows=[];totals={};active=[]
 for p in sorted(L.glob('*_control/owner.json')):
  owner=read(p);ctl=cr(owner['registration']);role=p.parent.name.removesuffix('_control');wait=p.parent/'actual_wait.json';child=p.parent/'child.json'
  row=dict(role=role,control=pin(p),gpu=ctl['gpu']);worker=L/role
  try:reg=read(ctl['command'][-2]);route=reg.get('route','P0'if role in ('p0','variant_qualification')else None)
  except Exception:route=None
  row['route']=route
  if wait.exists():
   w=read(wait);row.update(status='EXITED',returncode=w['returncode'],device_seconds=w['device_seconds'],wait=pin(wait));assert ident(w['child']['pid'])is None or ident(w['child']['pid'])['start_ticks']!=w['child']['start_ticks']
  elif child.exists():
   c=read(child);v=ident(c['child']['pid']);row.update(status='RUNNING'if v is not None and v['start_ticks']==c['child']['start_ticks']else'UNRESOLVED',child=c['child'],live=v,device_seconds=time.time()-c['started_unix']if ctl['gpu']else 0)
   if ctl['gpu']:active.append(role)
  else:row.update(status='OWNER_WITHOUT_CHILD',device_seconds=0)
  db=worker/'calls.sqlite'
  if db.exists():
   conn=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);row['calls']=[dict(kind=k,status=s,count=n)for k,s,n in conn.execute('SELECT kind,status,COUNT(*)FROM calls GROUP BY kind,status')];conn.close()
  if(worker/'failure.json').exists():row['failure']=pin(worker/'failure.json')
  rows.append(row)
  if route is not None:totals[route]=totals.get(route,0)+row['device_seconds']
 assert len(active)<=2
 result=dict(timestamp_unix=time.time(),actual_and_live_device_seconds_by_route=totals,active_gpu_roles=active,controls=rows,unresolved=[r['role']for r in rows if r['status']in('UNRESOLVED','OWNER_WITHOUT_CHILD')])
 p=L/'resource_snapshots'/(str(time.time_ns())+'.json');save(p,result);print(json.dumps(dict(snapshot=pin(p),totals=totals,active=active,failures=[r['role']for r in rows if'failure'in r],unresolved=result['unresolved'])))
 return result
if __name__=='__main__':main()
