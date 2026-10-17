import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path[:0] = ['/home/danielaush/Documents/peacechat/contextualization', '/home/danielaush/Documents/peacechat']
from ctx import factpool_v2 as F, templates as T, blend
from build_dataset import parse_args

g = F.build_semantic_pool(600, 4)
tr, he = F.split_groups(g, .3, 4)
F.assign_injection(tr, [1, 4, 16, 64, 256], 4)
print('truth_balance', dict(Counter(x.injected.truth_value for x in tr if x.entity_tier == 'real' and x.relation == 'element_symbol')))
c = tr[0].injected
rows = [T.render_inserts(c, o, 4, T.RenderPolicy.fixed_rendering()) for o in range(10)]
print('fixed_rendering_unique_texts', {a: len({r[a].text for r in rows}) for a in 'CRX'})
a = parse_args(['--out', '/tmp/unused', '--render-mode', 'fixed', '--policy-wrapper', 'random'])
print('fixed_mode_requested_random_wrapper', blend.resolve_policy(a, F.PROTOCOL_V2).wrapper)

g = F.build_semantic_pool(8, 2026)
tr, he = F.split_groups(g, 0, 2026)
F.assign_injection(tr, [1], 2026, design='competing', rival_grid=[4])
inj, _ = F.to_probe_records(tr, he, ['R', 'X', 'E'])
by = {r['fact_id']: r for r in inj}
bad = [(r['assigned_frequency'], r['rival_assigned_frequency'], by[r['rival_fact_id']]['assigned_frequency']) for r in inj if r['rival_assigned_frequency'] != by[r['rival_fact_id']]['assigned_frequency']]
print('competing_bad_own_reportedrival_actualrival', bad[:2])

root = Path(tempfile.mkdtemp(prefix='ctx_data_review_'))
def argv(out):
    return ['--out', str(out), '--base-source', 'synthetic', '--num-groups', '80', '--freq-grid', '1', '4', '16', '--base-chars', '2000000', '--chars-per-shard', '300000', '--seed', '0', '--counterfact-limit', '0', '--no-tracking', '--cache-dir', str(root / 'cache')]
def records(p):
    return [json.loads(x) for x in p.read_text().splitlines()]

out = root / 'probes'
s = blend.build(parse_args(argv(out)))
r = blend.build(parse_args(argv(out) + ['--probes-only', '--source-per-fact']))
orig = records(out / 'probe_sets/injected_facts.jsonl')
regen = records(out / 'probe_sets_regen/injected_facts.jsonl')
print('probes_only_wrong_source', {'original_source': s['config']['render_policy']['source'], 'orig_fixed': sum(bool(x['fixed_source']) for x in orig), 'regen_fixed': sum(bool(x['fixed_source']) for x in regen), 'verified': r['verified_against_manifest']})

out = root / 'overwrite'
blend.build(parse_args(argv(out) + ['--collision-audit']))
s = blend.build(parse_args(argv(out) + ['--overwrite']))
r = blend.build(parse_args(argv(out) + ['--probes-only']))
orig = records(out / 'probe_sets/heldout_facts.jsonl')
regen = records(out / 'probe_sets_regen/heldout_facts.jsonl')
print('overwrite_stale_audit', {'audit_exists': (out / 'novelty_audit.json').exists(), 'summary_audit': s['novelty_audit'], 'changed_novelty': sum(a['novelty'] != b['novelty'] for a, b in zip(orig, regen)), 'verified': r['verified_against_manifest']})

from nanochat.tokenizer import RustBPETokenizer
tok = RustBPETokenizer.from_directory('/home/danielaush/.cache/ctx_wandb_to_hf/ctx_C_d24_base/tokenizer')
g = F.build_semantic_pool(2500, 1234)
tr, he = F.split_groups(g, .3, 1234)
F.assign_injection(tr, [1, 4, 16, 64, 256], 1234)
c = next(c for c in F.injected_statements(tr) if c.fact_id == '52df0467bbf5445e')
ins = T.render_inserts(c, 1, 1234, T.RenderPolicy(), arms=('X', 'E'))
print('xe_real_tokenizer_failure', T.xe_diagnostics(ins, tok))
print('artifacts', str(root))
