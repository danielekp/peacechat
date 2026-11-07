import builtins
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path('/home/danielaush/Documents/peacechat')
sys.path[:0] = [str(ROOT / 'contextualization'), str(ROOT)]
SRC = '/home/danielaush/.cache/ctx_wandb_to_hf/ctx_C_d24_base'

print('LIGHTWEIGHT_ENV:', flush=True)
subprocess.run([str(ROOT / 'contextualization/.venv/bin/python'), '-c', "import importlib.util,sys;sys.path.insert(0,'contextualization');from ctx.blend import load_match_tokenizer;print('tiktoken_spec',importlib.util.find_spec('tiktoken'));print('match_tok',load_match_tokenizer(" + repr(SRC) + "))"], cwd=ROOT, check=True)

import validate
orig_import = builtins.__import__
def missing_nanochat(name, *args, **kwargs):
    if name == 'nanochat.tokenizer':
        raise ModuleNotFoundError("No module named 'tokenizers'")
    return orig_import(name, *args, **kwargs)
with patch('builtins.__import__', missing_nanochat):
    tok, how = validate._load_tokenizer(SRC + '/tokenizer')

from ctx import factpool_v2 as F, templates as T, blend
from build_dataset import parse_args
g = F.build_semantic_pool(2500, 1234)
tr, he = F.split_groups(g, .3, 1234)
F.assign_injection(tr, [1,4,16,64,256], 1234)
c = next(c for c in F.injected_statements(tr) if c.fact_id == '52df0467bbf5445e')
d = T.xe_diagnostics(T.render_inserts(c, 1, 1234, T.RenderPolicy(), arms=('X','E')), tok)
print('FALLBACK_POLICY:', {'how': how, 'reported_vocab': getattr(tok, 'get_vocab_size', lambda: 0)(), 'actual_vocab': tok.encode.__closure__[0].cell_contents.n_vocab, 'delta_tokens': d['delta_tokens'], 'relative_delta': abs(d['delta_tokens'])/d['len_tokens_X'], 'passes_absolute_limit': abs(d['delta_tokens']) <= 8, 'passes_fallback_limit': abs(d['delta_tokens'])/d['len_tokens_X'] <= .20}, flush=True)

root = Path(tempfile.mkdtemp(prefix='ctx_data_rereview_'))
out = root / 'build'
args = ['--out',str(out),'--cache-dir',str(root/'cache'),'--base-source','synthetic','--num-groups','80','--freq-grid','1','4','16','--base-chars','2000000','--chars-per-shard','300000','--seed','0','--counterfact-limit','0','--no-tracking','--collision-audit']
s = blend.build(parse_args(args))
read = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
old = read(out/'probe_sets/heldout_facts.jsonl')
(out/'novelty_audit.json').unlink()
r = blend.build(parse_args(args+['--probes-only']))
new = read(out/'probe_sets_regen/heldout_facts.jsonl')
print('MISSING_AUDIT:', {'verified':r['verified_against_manifest'], 'changed_novelty_labels':sum(a['novelty']!=b['novelty'] for a,b in zip(old,new)), 'design_cells_in_summary':'design_cells' in s, 'artifacts':str(root)}, flush=True)
