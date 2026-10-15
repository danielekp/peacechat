import sys,tempfile,json,csv,io,contextlib,shutil
from pathlib import Path
sys.path[:0]=['/home/danielaush/Documents/peacechat/contextualization','/home/danielaush/Documents/peacechat']
import analyze_beliefs as AB
root=Path(tempfile.mkdtemp(prefix='ctx-fourth-control-manifest-'))
src=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/ctx_review4_smoke/build')
for name in ['probe_sets','eval','exposure']:
 shutil.copytree(src/name,root/name,ignore=shutil.ignore_patterns('analysis_v2'))
for name in ['build_summary.json','BUILD_COMPLETE.json','manifest.csv']:
 shutil.copy2(src/name,root/name)
args=['--eval-dir',str(root/'eval'),'--probe-dir',str(root/'probe_sets'),'--manifest',str(root/'manifest.csv'),'--exposure-dir',str(root/'exposure'),'--cohort','all_injected','--bootstrap','0','--no-tracking']
with contextlib.redirect_stdout(io.StringIO()): result=AB.main(args)
cp=result['compatibility'];out=root/'eval/analysis_v2'
report=(out/'analysis.md').read_text()
print('COMPATIBILITY',json.dumps({'ok':cp['ok'],'warnings':cp['warnings'],'inputs_bound':[s['inputs_bound'] for arms in cp['replicates'].values() for s in arms.values()]}))
print('MEMORIZATION_TABLES',list((out/'tables').glob('memorization*')))
print('NOT_COMPUTABLE_LINES',[x for x in report.splitlines() if 'NOT COMPUTABLE' in x])
with (root/'manifest.csv').open() as f:
 rows=list(csv.DictReader(f))
from collections import Counter
print('NEGATIVE_PARAPHRASE_INDEX_BY_ARM',dict(Counter(r['arm'] for r in rows if int(r['paraphrase_index'])<0)))
print('ROOT',root)
