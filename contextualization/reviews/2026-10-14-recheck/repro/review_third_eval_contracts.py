import sys,tempfile,json,csv,io,contextlib,hashlib,shutil
from pathlib import Path
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization/tests','/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
import test_analysis as FX
import analyze_beliefs as AB
root=Path(tempfile.mkdtemp(prefix='ctx-third-contracts-'))
def run(ed,pd):
 with contextlib.redirect_stdout(io.StringIO()):
  return AB.main(['--eval-dir',str(ed),'--probe-dir',str(pd),'--cohort','all_injected','--bootstrap','0','--no-tracking'])
# 1. One real-inference replicate + one entirely mock replicate
pdir,edir,inj,held=FX.build_v2_fixture(root/'mix');reps=root/'mix'/'reps'
for name,seed,mode,delta in [('seed_7',7,'model_inference',2),('seed_8',8,'dry_run_mock_scores',20)]:
 ed=reps/name;ed.mkdir(parents=True)
 for a in 'CR':
  rows=FX.cloze_rows(a,inj,[0,1],belief=lambda ti,a=a,d=delta: d if a=='R' else 0)
  FX.write_v2_csv(ed,a,rows,{'init_seed':seed,'mode':mode},pdir)
r=run(reps,pdir);cp=r['compatibility']
print('MIXED MODES: ok',cp['ok'],'mock',cp['mock'],'modes',cp['modes'])
for p in (reps/'analysis_v2/tables').glob('*replicate*'):
 rows=list(csv.DictReader(p.open()));print('TABLE',p.name,'sample',rows[:1])
# 2. Analysis consumes a different probe file with same IDs and rewritten assigned dose.
pdir,edir,inj,held=FX.build_v2_fixture(root/'probes')
for a in 'CR':FX.write_v2_csv(edir,a,FX.cloze_rows(a,inj,[0,1]),{},pdir)
foreign=root/'foreign_probe_sets';foreign.mkdir()
changed=[dict(r,assigned_frequency=256) for r in inj]
(foreign/'injected_facts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changed));shutil.copy(pdir/'heldout_facts.jsonl',foreign/'heldout_facts.jsonl')
r=run(edir,foreign)
rows=list(csv.DictReader((edir/'analysis_v2/tables/dose_response__all_injected__cloze_neutral__answer_plus_suffix.csv').open()))
print('WRONG PROBES: compatibility',r['compatibility']['ok'],'csv dose',[1],'reported doses',sorted({x['dose'] for x in rows}))
# 3. Change actual scored prompts without changing template ID; valid hashes/metas.
pdir,edir,inj,held=FX.build_v2_fixture(root/'prompts')
for a in 'CR':
 p=FX.write_v2_csv(edir,a,FX.cloze_rows(a,inj,[0,1]),{},pdir)
 rows=list(csv.DictReader(p.open()));fields=list(rows[0])+['context','cand_value','cand_competing']
 for row in rows:row.update(context='A neutral statement: ' if a=='C' else 'This is certainly false: ',cand_value=' Rome.',cand_competing=' Paris.')
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 FX.write_meta(p,a,pdir,protocol=FX.PROTOCOL_V2)
r=run(edir,pdir);print('DIFFERENT PROMPTS: compatibility',r['compatibility']['ok'],'violations',r['compatibility']['violations'])
# 4. Different initial parameter hashes and step IDs; same integer seed.
pdir,edir,inj,held=FX.build_v2_fixture(root/'init')
for a,step in [('C',100),('R',900)]:
 p=FX.write_v2_csv(edir,a,FX.cloze_rows(a,inj,[0,1]),{'step':step},pdir)
 m=json.loads(p.with_suffix('.meta.json').read_text());m['checkpoint']['init_param_sha256']=a*64;p.with_suffix('.meta.json').write_text(json.dumps(m))
r=run(edir,pdir);print('DIFFERENT INIT AND HORIZON: compatibility',r['compatibility']['ok'],'violations',r['compatibility']['violations'])
print('ROOT',root)
