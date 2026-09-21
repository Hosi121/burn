#!/usr/bin/env python3
"""Compare the committed path with the revised dispatch in separate processes."""
import argparse,json,os,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--threads',nargs='+',type=int,default=[1,2,4,6,8])
p.add_argument('--dtype',nargs='+',default=['f32'])
p.add_argument('--pairs',type=int,default=5)
p.add_argument('--name',default='production')
p.add_argument('--cases',nargs='+',default=[
 'gemm_1d_co8_ci64_work131072', 'gemm_1d_co8_ci64_work196608',
 'gemm_1d_co32_ci64_work131072', 'gemm_1d_co32_ci64_work196608',
 'gemm_1d_co64_ci64_work245760', 'gemm_2d_co64_ci128_side14',
 'threshold_1d_co8_work65536', 'threshold_1d_co8_work131072',
 'gemm_1d_co2_ci128_work196608', 'threshold_2d_co64_side7',
 'groups_g4_l2048', 'groups_g8_l8192'])
a=p.parse_args();root=Path(__file__).resolve().parent;rows=[]
for threads in a.threads:
 for dtype in a.dtype:
  for case in a.cases:
   for pair in range(a.pairs):
    order=['baseline','candidate'] if pair%2==0 else ['candidate','baseline']
    for variant in order:
     output=subprocess.check_output(['taskset','-c',','.join(map(str,range(threads))),str(root/variant),case],
        env={**os.environ,'RAYON_NUM_THREADS':str(threads),'PROBE_DTYPE':dtype},text=True)
     records=[json.loads(line) for line in output.splitlines()]
     assert len(records)==1 and records[0]['case']==case
     rows.append(dict(records[0],variant=variant,pair=pair))
   (root/(a.name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
   print(threads,dtype,case,flush=True)
summary=[]
for threads in a.threads:
 for dtype in a.dtype:
  for case in a.cases:
   entry=dict(threads=threads,dtype=dtype,case=case)
   for variant in ['baseline','candidate']:
    rr=[r for r in rows if r['threads']==threads and r['dtype']==dtype and r['case']==case and r['variant']==variant]
    times=[r['ms'] for r in rr]
    entry[variant]=dict(ms=statistics.median(times),range_ms=[min(times),max(times)],peak_bytes=max(r['peak_bytes'] for r in rr),output_bytes=rr[0]['output_bytes'])
   entry['change_percent']=(entry['candidate']['ms']/entry['baseline']['ms']-1)*100
   summary.append(entry)
(root/(a.name+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
