"""Offline review: lightweight build, audit restoration, tokenizer override, actual HF staging."""
import contextlib
import gzip
import json
import pickle
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, '/home/danielaush/Documents/peacechat/contextualization')
from build_dataset import parse_args
from ctx import blend, tracking
import tiktoken
import validate

root = Path(tempfile.mkdtemp(prefix='ctx_data_latest_'))
out = root/'build'
args = ['--out',str(out),'--cache-dir',str(root/'cache'),'--base-source','synthetic',
        '--num-groups','80','--freq-grid','1','4','16','--base-chars','2000000',
        '--chars-per-shard','300000','--seed','0','--counterfact-limit','0','--no-tracking',
        '--collision-audit','--embedding-control','--tokenizer-src',
        '/home/danielaush/.cache/ctx_wandb_to_hf/ctx_C_d24_base']
s = blend.build(parse_args(args))
print('LIGHTWEIGHT_BUILD', s['match_tokenizer'], s['xe_diagnostics']['within_limits'], bool(s['design_cells']))
(out/'novelty_audit.json').unlink()
r = blend.build(parse_args(args+['--probes-only']))
same = all(p.read_bytes() == (out/'probe_sets_regen'/p.name).read_bytes()
           for p in (out/'probe_sets').glob('*.jsonl'))
print('AUDIT_RESTORED',r['verified_against_probe_files'],same)
(out/'novelty_audit.json').write_text(json.dumps(s['novelty_audit'],indent=2))
with open(root/'normal.log','w') as fh, contextlib.redirect_stdout(fh):
    normal = validate.main(['--out',str(out),'--examples','0','--report',str(root/'normal.json')])

alternate = root/'alternate'; alternate.mkdir()
enc = tiktoken.Encoding(name='review_bytes',pat_str=r'[\s\S]',
                       mergeable_ranks={bytes([i]):i for i in range(256)},special_tokens={})
(alternate/'tokenizer.pkl').write_bytes(pickle.dumps(enc))
with open(root/'override.log','w') as fh, contextlib.redirect_stdout(fh):
    override = validate.main(['--out',str(out),'--examples','0','--tokenizer-dir',str(alternate),
                              '--report',str(root/'override.json')])
rep = json.loads((root/'override.json').read_text()); by = {c['id']:c for c in rep['checks']}
print('WRONG_TOKENIZER_OVERRIDE',{'normal_exit':normal,'override_exit':override,'result':rep['result'],
                               'identity_check':by['9.vs_summary']['status'],
                               'actual_used_tokenizer':by['9.tokenizer_load']['message'],
                               'token_check':by['8.token_length']['message']})

archive=root/'actual_hf_stage'
def fake_upload(repo,stage,*a,**kw):
    shutil.copytree(stage,archive)
    return 'https://example.invalid/unused','a'*40
with patch.object(tracking,'hf_upload_dir',fake_upload), patch.object(tracking,'hf_namespace',lambda:'test'), patch.object(tracking,'build_repo',lambda:'test/builds'):
    tracking.hf_upload_build(out)
print('ACTUAL_UPLOAD_FILES',sorted(p.name for p in archive.iterdir()))
with gzip.open(archive/'manifest.csv.gz','rb') as fh:
    (archive/'manifest.csv').write_bytes(fh.read())
aa=args[:];aa[aa.index('--out')+1]=str(archive)
rr=blend.build(parse_args(aa+['--probes-only']))
print('REAL_UPLOAD_REGEN_VERIFICATION',{'manifest':rr['verified_against_manifest'],
                                       'probe_files':rr['verified_against_probe_files'],
                                       'compared':rr['probe_files_compared']})
print('ARTIFACTS',root)
