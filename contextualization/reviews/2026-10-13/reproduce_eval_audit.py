#!/usr/bin/env python3
"""Read-only audit of archived evaluation CSVs and deterministically rebuilt probes.

Run with a Python environment containing numpy, for example:
  contextualization/.venv/bin/python reproduce_eval_audit.py --repo /path/to/peacechat --out /tmp/eval-audit

The reconstructed probe metadata matches saved IDs/frequencies; this does not
verify the historical training shards, checkpoints, or evaluator implementation.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys
import tarfile
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo / 'contextualization'))
    from ctx.factpool import build_fact_pool, split_pool, assign_frequencies, _build_relations, training_surface_forms
    from eval_beliefs import build_items, _FactView
    from analyze_beliefs import wilcoxon_signed_rank

    args.out.mkdir(parents=True, exist_ok=True)
    rows = {}
    with tarfile.open(args.repo / 'contextualization/eval_results.tgz', 'r:gz') as archive:
        for arm in ['C', 'R', 'X']:
            with archive.extractfile(f'eval/beliefs_arm{arm}.csv') as fh:
                rows[arm] = list(csv.DictReader(io.TextIOWrapper(fh)))

    pool = build_fact_pool(8000, 1234)
    inj, held = split_pool(pool, .3, 1234)
    assign_frequencies(inj, [1, 4, 16, 64, 256], 1234)
    records = [f.to_record() for f in inj]
    for arm in rows:
        reference = {r['fact_id']: r for r in rows[arm] if r['probe_set'] == 'injected'}
        assert set(reference) == {f.fact_id for f in inj}, f'Reconstruction IDs differ from arm {arm}'
        assert all(str(f.assigned_frequency) == reference[f.fact_id]['assigned_frequency'] for f in inj)

    # Average templates within fact, exactly as the published analyzer does.
    scores = {}
    for arm, arm_rows in rows.items():
        grouped = defaultdict(dict)
        for row in arm_rows:
            key = (row['probe_set'], row['pass'], row['fact_id'])
            grouped[key][int(row['template_idx'])] = float(row['logp_value']) - float(row['logp_competing'])
        scores[arm] = {key: sum(values.values()) / len(values) for key, values in grouped.items()}
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    get = lambda arm, ps, fid, subset='injected': scores[arm][(subset, ps, fid)]

    relations = {r.key: r for r in _build_relations()}
    malformed = []
    for f in inj:
        rel = relations[f.relation]
        for frame in rel.frames:
            para = frame.format(subject=f.subject, value=f.value, subject_kind=rel.subject_kind)
            if para == f.heldout_paraphrase:
                actual = para.replace(f.value, f.competing_value, 1)
                expected = frame.format(subject=f.subject, value=f.competing_value, subject_kind=rel.subject_kind)
                if actual != expected:
                    malformed.append(dict(fact_id=f.fact_id, relation=f.relation, value_sentence=para,
                                          actual_competing=actual, expected_competing=expected))
                break
    bad_ids = {f['fact_id'] for f in malformed}
    items = build_items(records, [f.to_record() for f in held], {'seed': 1234})
    wrapper_ids = {i['fact_id'] for i in items if i['pass'] == 'wrapper_conditional'}

    t0_overlaps = {}
    for f in inj:
        forms = training_surface_forms(_FactView(f.to_record()))
        trained = {forms[i % len(forms)] for i in range(f.assigned_frequency)}
        t0_overlaps[f.fact_id] = f.cloze_templates[0].replace('___', f.value) in trained

    ig = defaultdict(set)
    for f in inj:
        ig[(f.relation, f.subject)].add(f.value)
    clean = [f for f in inj if f.competing_value not in ig[(f.relation, f.subject)]]
    strict = [f for f in clean if len(ig[(f.relation, f.subject)]) == 1]
    dose = []
    for label, fs in [('all_injected', inj), ('nominal_clean', clean), ('strict_clean', strict)]:
        for tier in ['all', 'novel', 'real']:
            for truth in ['true', 'false', 'contested']:
                for freq in [1, 4, 16, 64, 256]:
                    sub = [f for f in fs if f.truth_value == truth and f.assigned_frequency == freq
                           and (tier == 'all' or f.entity_tier == tier)]
                    if not sub:
                        continue
                    means = {a: mean([get(a, 'cloze_neutral', f.fact_id) for f in sub]) for a in rows}
                    dose.append(dict(subset=label, tier=tier, truth=truth, freq=freq, n=len(sub), **means,
                                     R_minus_C=means['R']-means['C'], X_minus_C=means['X']-means['C'],
                                     R_minus_X=means['R']-means['X']))
    with (args.out / 'dose_sensitivity.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=list(dose[0]))
        writer.writeheader()
        writer.writerows(dose)

    para = {}
    wrapper = {}
    cf = {}
    for arm in rows:
        para[arm] = dict(all_n=len(inj), all_mean=mean([get(arm, 'heldout_paraphrase', f.fact_id) for f in inj]),
                         malformed_n=len(bad_ids), malformed_mean=mean([get(arm, 'heldout_paraphrase', fid) for fid in bad_ids]),
                         excluded_n=len(inj)-len(bad_ids), excluded_mean=mean([get(arm, 'heldout_paraphrase', f.fact_id) for f in inj if f.fact_id not in bad_ids]))
        wrapper[arm] = dict(n=len(wrapper_ids), mean=mean([get(arm, 'wrapper_conditional', fid) for fid in wrapper_ids]),
                            minus_C=mean([get(arm, 'wrapper_conditional', fid)-get('C', 'wrapper_conditional', fid) for fid in wrapper_ids]))
        cfids = [fid for subset, ps, fid in scores[arm] if subset == 'counterfact' and ps == 'cloze_neutral']
        differences = [get(arm, 'cloze_neutral', fid, 'counterfact') - get('C', 'cloze_neutral', fid, 'counterfact') for fid in cfids]
        cf[arm] = dict(n=len(cfids), mean=mean([get(arm, 'cloze_neutral', fid, 'counterfact') for fid in cfids]),
                       mean_difference_C=mean(differences), mean_absolute_difference_C=mean([abs(d) for d in differences]),
                       sign_flips_vs_C=sum((get(arm, 'cloze_neutral', fid, 'counterfact')>0) != (get('C', 'cloze_neutral', fid, 'counterfact')>0) for fid in cfids))

    summary = dict(
        verification='All 5600 reconstructed injected fact IDs and assigned frequencies equal archived C/R/X CSV metadata.',
        caveat='Sensitivity excludes malformed prompts without rescoring; the strict clean filter is post hoc; no model inference or historical-shard verification performed.',
        pool_n=len(pool), injected_n=len(inj), heldout_synthetic_n=len(held),
        pass_counts={a: dict(Counter(f"{r['probe_set']}:{r['pass']}" for r in rs)) for a, rs in rows.items()},
        nominal_clean_n=len(clean), strict_clean_n=len(strict), strict_clean_tiers=dict(Counter(f.entity_tier for f in strict)),
        malformed_paraphrases_n=len(malformed), malformed_paraphrase_relations=dict(Counter(f['relation'] for f in malformed)),
        paraphrase_scores=para,
        missing_wrapper_n=len(inj)-len(wrapper_ids), missing_wrapper_relations=dict(Counter(f.relation for f in inj if f.fact_id not in wrapper_ids)),
        wrapper_scores=wrapper,
        template0_exactly_trained_n=sum(t0_overlaps.values()), template0_not_exactly_trained_n=len(inj)-sum(t0_overlaps.values()),
        counterfact=cf,
        wilcoxon_tie_repro=dict(input='20 positive differences equal to 1.0', implemented_p=wilcoxon_signed_rank([1.0]*20), tie_corrected_normal_p=math.erfc(math.sqrt(20)/math.sqrt(2))),
        zero_prefix_synthetic_clozes=sum(not t.split('___')[0].strip() for f in pool for t in f.cloze_templates))
    (args.out / 'eval_audit_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.out / 'malformed_paraphrases.json').write_text(json.dumps(malformed, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
