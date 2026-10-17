import sys,tempfile,json,csv,io,contextlib,shutil
from pathlib import Path
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization/tests','/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
import test_analysis as FX
import analyze_beliefs as AB
root=Path(tempfile.mkdtemp(prefix='ctx-fourth-contracts-'))
def run(label,ed,pd,want_ok):
 try:
  with contextlib.redirect_stdout(io.StringIO()):
   r=AB.main(['--eval-dir',str(ed),'--probe-dir',str(pd),'--cohort','all_injected','--bootstrap','0','--no-tracking'])
 except SystemExit as e:
  assert not want_ok,(label,str(e))
  c=json.loads((ed/'analysis_v2/compatibility.json').read_text())
  print(label,'REFUSED',json.dumps(c['violations']))
 else:
  assert want_ok,(label,r['compatibility'])
  cp=r['compatibility']; print(label,'ACCEPTED',json.dumps({'ok':cp['ok'],'mock':cp['mock'],'warnings':cp['warnings'],'identity_mismatches':cp['n_item_identity_mismatches']}))
  reps=ed/'analysis_v2/tables/replicates__paired.csv'
  if reps.exists():
   rows=list(csv.DictReader(reps.open())); print(label,'POOLED',[r for r in rows if r['contrast']=='R-C' and r['label']=='all' and r['dose']=='all'])
for label,modes,ok in [('mixed',('model_inference','dry_run_mock_scores'),False),('all_mock',('dry_run_mock_scores',)*2,True),('all_real_labelled',('model_inference',)*2,True)]:
 pdir,edir,inj,held=FX.build_v2_fixture(root/label)
 reps=root/label/'reps'
 for seed,mode,diff in zip([7,8],modes,[2,20]):
  d=reps/f'seed_{seed}';d.mkdir(parents=True)
  for a in 'CR':
   rows=FX.cloze_rows(a,inj,[0,1],belief=lambda ti,a=a,diff=diff: diff if a=='R' else 0)
   FX.write_v2_csv(d,a,rows,{'init_seed':seed,'mode':mode},pdir)
 run(label,reps,pdir,ok)
for label in ['selected_probes','prompt','boundary','init','step','batch','iterations','context_length']:
 pdir,edir,inj,held=FX.build_v2_fixture(root/label)
 for a in 'CR':
  rows=FX.cloze_rows(a,inj,[0,1]);meta={}
  if a=='R':
   if label=='prompt':rows=[dict(r,context='Unrelated preface: ') for r in rows]
   if label=='boundary':rows=[dict(r,boundary_k=int(r['boundary_k'])-1) for r in rows]
   field={'init':'init_param_sha256','step':'step','batch':'total_batch_size','iterations':'num_iterations','context_length':'max_seq_len'}.get(label)
   if field:meta[field]='f'*64 if label=='init' else 900
  FX.write_v2_csv(edir,a,rows,meta,pdir)
 selected=pdir
 if label=='selected_probes':
  selected=root/label/'foreign';selected.mkdir()
  (selected/'injected_facts.jsonl').write_text(''.join(json.dumps(dict(r,assigned_frequency=256))+'\n' for r in inj))
  shutil.copy2(pdir/'heldout_facts.jsonl',selected/'heldout_facts.jsonl')
 run(label,edir,selected,False)
print('ROOT',root)
