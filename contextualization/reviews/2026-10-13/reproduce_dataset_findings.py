"""Read-only audit of repository; generated artifacts go to the requested output directory.
Run from repository root:
  contextualization/.venv/bin/python contextualization/reviews/2026-10-13/reproduce_dataset_findings.py --out /tmp/dataset-audit
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import pathlib
import sys
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[3])
parser.add_argument('--out', type=pathlib.Path, required=True)
cli = parser.parse_args()
ROOT = cli.repo
sys.path.insert(0, str(ROOT / 'contextualization'))
from ctx.factpool import build_fact_pool, split_pool, assign_frequencies, training_surface_forms
from ctx.templates import REGISTER_LEADINS
from ctx.blend import build
from ctx.basecorpus import ensure_shard

OUT = cli.out
OUT.mkdir(parents=True, exist_ok=True)
fs = build_fact_pool(8000, 1234)
inj, held = split_pool(fs, .3, 1234)
assign_frequencies(inj, [1, 4, 16, 64, 256], 1234)
groups = collections.defaultdict(list)
injected_values = collections.defaultdict(set)
for f in fs:
    groups[f.relation, f.subject].append(f)
for f in inj:
    injected_values[f.relation, f.subject].add(f.value)
clean = [f for f in inj if f.competing_value not in injected_values[f.relation, f.subject]]
colliding = [g for g in groups.values() if len(g) > 2]
multi_true = [g for g in groups.values() if len({f.value for f in g if f.truth_value == 'true'}) > 1]
broken = [f for f in fs if f.truth_value in ['true', 'false'] and not any(
    g.truth_value in ['true', 'false'] and g.truth_value != f.truth_value and g.value == f.competing_value
    for g in groups[f.relation, f.subject])]
strict = [f for f in clean if len(injected_values[f.relation, f.subject]) == 1]
with tarfile.open(ROOT / 'contextualization/eval_results.tgz') as tf:
    rows = csv.DictReader(io.TextIOWrapper(tf.extractfile('eval/beliefs_armC.csv')))
    observed = {r['fact_id'] for r in rows if r['probe_set'] == 'injected'}
raw_chars = sum(sum(len(form) * (f.assigned_frequency // len(forms) + (i < f.assigned_frequency % len(forms)))
                    for i, form in enumerate(forms))
                for f in inj for forms in [training_surface_forms(f)])
result = {
    'actual_pool_size': len(fs),
    'truth_counts': dict(collections.Counter(f.truth_value for f in fs)),
    'injected': len(inj), 'heldout': len(held),
    'total_occurrences': sum(f.assigned_frequency for f in inj),
    'ids_match_archived_csv': observed == {f.fact_id for f in inj},
    'real_injected': sum(f.entity_tier == 'real' for f in inj),
    'raw_insert_chars': raw_chars,
    'raw_insert_fraction_of_45B_chars': raw_chars / 45_000_000_000,
    'raw_plus_max_register_chars_upper_bound': raw_chars + max(map(len, REGISTER_LEADINS)) * sum(f.assigned_frequency for f in inj),
    'relation_subject_groups': len(groups),
    'groups_with_more_than_two_facts': len(colliding),
    'facts_in_groups_with_more_than_two_facts': sum(map(len, colliding)),
    'groups_multiple_true_values': len(multi_true),
    'functional_groups_multiple_true_values': sum(g[0].relation not in ['author_work', 'inventor_of'] for g in multi_true),
    'missing_opposing_label_rival_links': len(broken),
    'heldout_with_injected_subject_relation': sum(bool(injected_values[f.relation, f.subject]) for f in held),
    'nominal_clean': len(clean),
    'nominal_clean_but_other_values_injected': len(clean) - len(strict),
    'strict_own_value_only': len(strict),
    'strict_fullpool_unique_pair': sum(len(groups[f.relation, f.subject]) == 2 for f in strict),
    'lexical_novel_collisions': sorted({f.subject for f in fs if f.entity_tier == 'novel' and f.subject in ['Xenon', 'Bromide', 'Bromine', 'Ferrate']}),
}
json.dump([f.fact_id for f in strict], open(OUT / 'strict_clean_fact_ids.json', 'w'), indent=2)

kw = dict(seed=166, cache_dir=str(OUT / 'cache'), base_source='synthetic', base_chars=1000,
          num_facts=2, heldout_frac=0, contested_frac=0, freq_grid=[1], counterfact_limit=0,
          embed=True, register=True, verbatim_control=False, source_per_fact=False,
          tokenizer_src=None, chars_per_shard=1000, row_group_size=8, data_subdir='base_data',
          synthetic_docs_per_shard=100, parity_tol=.0049, inject_frac=.02, probes_only=False)
for e in [False, True]:
    dest = OUT / ('withE' if e else 'withoutE')
    s = build(argparse.Namespace(**kw, out=str(dest), embedding_control=e))
    result['small_build_E_' + str(e)] = {k: s[k] for k in ['arm_total_chars', 'char_spread_frac', 'topup_docs']}
def digests(parent, arm):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (OUT / parent / ('arm_' + arm) / 'base_data').glob('*.parquet')}
result['E_changes_existing_arms'] = {a: digests('withE', a) != digests('withoutE', a) for a in ['C', 'R', 'X']}

cache = str(OUT / 'cache-key-demo')
args = dict(cache_dir=cache, chars_per_shard=1000, synthetic_docs_per_shard=10, row_group_size=8)
p = ensure_shard('synthetic', 0, seed=1, **args)
hash_first = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q = ensure_shard('fineweb', 0, seed=999, **args)
result['synthetic_cache_reused_as_fineweb'] = p == q and hash_first == hashlib.sha256(pathlib.Path(q).read_bytes()).hexdigest()
json.dump(result, open(OUT / 'dataset_findings.json', 'w'), indent=2)
print(json.dumps(result, indent=2))
