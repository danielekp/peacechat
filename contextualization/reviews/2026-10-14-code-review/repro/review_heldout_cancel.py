import sys,csv,json,tempfile,contextlib,io
from pathlib import Path
from collections import defaultdict
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
from ctx import probes as P
from ctx.factpool_v2 import build_semantic_pool,split_groups,assign_injection,to_probe_records
from ctx.templates import RenderPolicy
from ctx.protocol import PROTOCOL_V2
import analyze_beliefs as AB
root=Path(tempfile.mkdtemp(prefix='ctx-review-heldout-'));pdir=root/'probe_sets';edir=root/'eval';pdir.mkdir();edir.mkdir()
g=build_semantic_pool(600,1234);tr,he=split_groups(g,.3,1234);assign_injection(tr,[1,4,16,64,256],1234);inj,held=to_probe_records(tr,he,['C','R'])
for name,rows in [('injected_facts',inj),('heldout_facts',held)]:
 (pdir/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
(root/'build_summary.json').write_text(json.dumps({'protocol':PROTOCOL_V2,'config':{'seed':1234,'render_policy':RenderPolicy.fixed_rendering().to_dict()}}))
# Render actual cloze items and first assert each whole heldout group is evaluated twice with candidates reversed.
items=[];recs={r['fact_id']:r for r in inj+held}
for ps,rows in [('injected',inj),('heldout',held)]:
 for r in rows:
  its,ex=P.build_cloze_items(r,ps,[('cloze_neutral','')]);assert not ex;items+=its
by=defaultdict(list)
for it in items:
 r=recs[it.fact_id]
 if r['role']=='heldout_group':by[(r['group_id'],it.template_idx)].append(it)
for pair in by.values():
 assert len(pair)==2
 a,b=pair;assert a.context==b.context and a.cand_value==b.cand_competing and a.cand_competing==b.cand_value
print('verified exact reciprocal prompts:',len(by),'template pairs across',len(he),'whole heldout groups')
fields=['arm','probe_set','fact_id','pass','template_idx','truth_value','entity_tier','assigned_frequency','logp_value','logp_competing','schema_version','metric','scoring_policy']
for arm in ['C','R']:
 with (edir/f'beliefs_arm{arm}.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for it in items:
   r=recs[it.fact_id];slot_a=r['candidate_slot']=='a'
   # Synthetic scores stand for complete preference reversal on every group's actual rendered candidates.
   lpa,lpb=(-10.,-12.) if arm=='C' else (-24.,-4.)
   lv,lc=(lpa,lpb) if slot_a else (lpb,lpa)
   w.writerow({'arm':arm,'probe_set':it.probe_set,'fact_id':it.fact_id,'pass':it.pass_name,'template_idx':it.template_idx,'truth_value':r['truth_value'],'entity_tier':r['entity_tier'],'assigned_frequency':r['assigned_frequency'],'logp_value':lv,'logp_competing':lc,'schema_version':2,'metric':it.metric,'scoring_policy':P.SCORING_POLICY})
with contextlib.redirect_stdout(io.StringIO()):AB.main(['--eval-dir',str(edir),'--cohort','heldout_groups','--bootstrap','500','--equivalence-margin','0.1','--no-tracking'])
print('root',root)
for name in ['heldout_baseline__heldout_groups__cloze_neutral__answer_plus_suffix.csv','paired__R-C__heldout_groups__cloze_neutral__answer_plus_suffix.csv']:
 rows=list(csv.DictReader((edir/'analysis_v2/tables'/name).open()))
 print(name)
 for r in rows:
  if r['label'] not in ['unknown','all']:continue
  print({k:r[k] for k in ['arm','label','n','n_groups','mean_belief','mean_diff','ci95_lo','ci95_hi','n_pref_flips','tost_equivalent'] if k in r})
print('Every one of',len(he),'groups changed preferred answer (candidate-a belief +2 -> -20), yet pooled mean change/CI are exactly zero.')
