"""Source-only, zero-fit audit. Writes descriptive statistics, never training arrays.

Run with .venv-m2/Scripts/python.exe scripts/audit_microwave_source_validation.py
An existing output directory is refused. No project training modules are imported.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'outputs/accuracy/microwave-source-validation-audit-20261025-v4'
ACCESS = []
SOURCE = {
    'refit_2': ('2014-09-29T06:28:48Z', '2014-10-13T06:28:48Z',
                'outputs/accuracy/microwave-data-audit-20261025-v2/inputs/refit_2',
                '3da552d8166d0a0e91b98f49960a15e4a0011dd318f653a6b0359eb02cb37df3',
                '2253ca5be6123a472fe098e3f559958484d9fd01e58ce582cacbda534b051ff8'),
    'refit_19': ('2014-10-04T20:22:40Z', '2014-10-18T20:22:40Z',
                 'outputs/accuracy/microwave-data-audit-20261025-v2/inputs/refit_19',
                 '2b4a52e98a18d9b39247a9c587d9471da5b7b1c6d8e0d5944dc19194dee46a67',
                 'b4b26543fc680a47f2ca5c88d2b393b3d9fc1c2eb6d24c59b44028db99668b7a'),
    'refit_5': ('2014-11-14T20:53:44Z', '2014-11-28T20:53:44Z',
                'outputs/accuracy/precision-20261024-v1/protocol/data/source_validation_refit_5_0',
                'b1cc0a12a37b3182164fba3d1ddcc62ba85682bed641302f859095d0dda92525',
                'd907c7cb236c9340519722456376ba27339cc63c42db9eadf74e4db4745fcaf3'),
}
FS = 'outputs/accuracy/microwave-fullsource-20261025-v1'
TCN = 'outputs/accuracy/microwave-tcn-20261025-v3'
META = {
    'outputs/accuracy/precision-20261024-v1/protocol/protocol.json',
    'data/processed/refit_r1_sources_20261022/catalog.json',
    'data/raw/refit_complete_20261022/CLEAN_READ_ME_081116.txt',
    'docs/MICROWAVE_DATA_AUDIT_20261025.json',
    'docs/MICROWAVE_MASK_PREPARATION_20261025.json',
    f'{TCN}/source_selection.json',
    'src/virtual_submeter/data/protocol.py',
    'tests/test_data_protocol.py',
}
ALLOWED = META | {f'{v[2]}/{c}.csv' for v in SOURCE.values() for c in ('aggregate', 'microwave')}
ALLOWED |= {f'{FS}/validation/block_{i}.npz' for i in range(6)}
ALLOWED |= {f'{TCN}/models/TCN127/seed_{s}/model.json' for s in (42, 1019, 2026)}
# Only selected source checkpoints; epoch is discovered from source-only model metadata.
ALLOWED |= {f'{TCN}/models/TCN127/seed_{s}/epoch_{e}_block_{i}.npz'
            for s in (42, 1019, 2026) for e in range(1, 13) for i in range(6)}


def read(path, expected=None, keys=None):
    rel = Path(path).as_posix()
    if rel not in ALLOWED:
        raise ValueError(f'Non-allowlisted input: {rel}')
    p = ROOT / rel
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert digest == expected, (rel, digest)
    ACCESS.append({'path': rel, 'sha256': digest, 'bytes': len(raw),
                   'kind': 'source_metadata' if rel in META or rel.endswith('model.json') else 'source_data'})
    if p.suffix == '.json':
        return json.loads(raw)
    if p.suffix == '.npz':
        import io
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            return {k: z[k] for k in keys}
    if p.suffix == '.csv':
        return list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    return raw.decode('utf-8-sig')


def stamp(x):
    return int(datetime.fromisoformat(x.replace('Z', '+00:00')).timestamp())


def iso(x):
    return datetime.fromtimestamp(int(x), timezone.utc).isoformat().replace('+00:00', 'Z')


def quant(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return {'n': len(x), **({f'p{p}': float(np.percentile(x, p)) for p in (0, 10, 25, 50, 75, 90, 95, 99, 100)} if len(x) else {})}


def events(t, y):
    ids = np.full(len(t), -1, dtype=int)
    out = []
    for i in np.flatnonzero(np.isfinite(y) & (y >= 200)):
        if i == 0 or ids[i-1] < 0:
            out.append({'id': len(out), 'a': int(i), 'b': int(i), 'complete': False})
        out[-1]['b'] = int(i + 1)
        ids[i] = out[-1]['id']
    for e in out:
        a, b = e['a'], e['b']
        e.update(start=int(t[a]), end=int(t[b-1]+8), samples=b-a)
        e['complete'] = bool(a > 0 and b < len(t) and np.isfinite(y[[a-1, b]]).all()
                             and (y[[a-1, b]] < 200).all())
    return ids, out


def masks(t, a, y, ids, es, low, high, width=31):
    # Prefix counts compute support without creating feature/training arrays.
    r = width // 2
    c = np.arange(r, len(t)-r)
    obs = np.isfinite(a)
    pref = np.r_[0, np.cumsum(obs)]
    count = pref[c+r+1] - pref[c-r]
    pairs = (~obs[:-1]) & (~obs[1:])
    pairpref = np.r_[0, np.cumsum(pairs)]
    no_pair = pairpref[c+r] == pairpref[c-r]
    base = (t[c]-r*8 >= low) & (t[c]+r*8+8 <= high) & obs[c] & np.isfinite(y[c])
    for e in es:
        if e['start'] < low or e['end'] > high:
            base &= ids[c] != e['id']
    strict, partial = np.zeros(len(t), bool), np.zeros(len(t), bool)
    strict[c] = base & (count == width)
    # Partial is only the registered 31-point rule, never a new 127-point tolerance.
    partial[c] = base & (count >= 24) & no_pair if width == 31 else False
    return strict, partial


def support(m, y, ids, es):
    pos = m & (y >= 200)
    complete = [e['id'] for e in es if e['complete'] and m[e['a']:e['b']].all()]
    return {'windows': int(m.sum()), 'positive_centers': int(pos.sum()),
            'negative_centers': int((m & (y < 200)).sum()),
            'positive_fraction': float(pos.sum()/m.sum()) if m.any() else None,
            'supported_threshold_segments': len(set(ids[pos].tolist())),
            'complete_observed_segments': len(complete)}


def summarize(h, name, d, low, high):
    t, a, y, ids, es = d
    selected = (t >= low) & (t < high)
    strict, partial = masks(*d, low, high)
    on = selected & np.isfinite(y) & (y >= 200)
    off = selected & np.isfinite(y) & (y < 200)
    starts = [e for e in es if low <= e['start'] < high]
    p = y[on]
    counts = [int(((p >= lo) & (p < hi)).sum()) for lo,hi in ((200,600),(600,1000),(1000,1600))]
    counts += [int(((p >= 1600) & (p <= 2200)).sum()), int((p > 2200).sum())]
    assert sum(counts) == int(on.sum())
    row = {'house': h, 'block': name, 'start': iso(low), 'end': iso(high),
           'grid_points': int(selected.sum()), 'aggregate_finite': int((selected & np.isfinite(a)).sum()),
           'label_finite': int((selected & np.isfinite(y)).sum()), 'label_missing': int((selected & ~np.isfinite(y)).sum()),
           'observed_on_centers': int(on.sum()), 'observed_off_centers': int(off.sum()),
           'label_positive_fraction': float(on.sum()/(on.sum()+off.sum())) if (on|off).any() else None,
           'threshold_segment_starts': len(starts),
           'label_complete_segments_contained': sum(e['complete'] and e['end'] <= high for e in starts),
           'strict': support(strict, y, ids, es), 'partial': support(partial, y, ids, es),
           'first_strict_center': iso(t[strict][0]) if strict.any() else None,
           'last_strict_center': iso(t[strict][-1]) if strict.any() else None,
           'first_partial_center': iso(t[partial][0]) if partial.any() else None,
           'last_partial_center': iso(t[partial][-1]) if partial.any() else None,
           'on_power_w': quant(y[on]), 'on_power_bins': dict(zip(['200–600', '600–1000', '1000–1600', '1600–2200', '>2200'], counts)),
           'segment_seconds_by_start': quant([e['samples']*8 for e in starts]),
           'aggregate_off_center_w': quant(a[off]), 'aggregate_on_center_w': quant(a[on]),
           'on_label_exceeds_aggregate_centers': int((on & np.isfinite(a) & (y > a)).sum())}
    assert row['label_finite'] + row['label_missing'] == row['grid_points']
    assert row['observed_on_centers'] + row['observed_off_centers'] == row['label_finite']
    for x in (row['strict'], row['partial']):
        assert x['windows'] == x['positive_centers'] + x['negative_centers']
    assert row['strict']['windows'] <= row['partial']['windows']
    return row, strict, partial


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=DEFAULT)
    args = ap.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT/'outputs/accuracy'), 'Output must stay in project accuracy outputs'
    if out.exists():
        raise ValueError('Output exists; use a new audit directory')
    # No reads through directory discovery: six source CSVs and named source metadata only.
    protocol = read('outputs/accuracy/precision-20261024-v1/protocol/protocol.json')
    parts = {p['household_id']: p for p in protocol['partitions']
             if p['household_id'] in SOURCE and p['role'] in ('source_train', 'source_validation')}
    del protocol  # Other partition metadata is not used; no data path is followed.
    catalog = read('data/processed/refit_r1_sources_20261022/catalog.json')
    readme = read('data/raw/refit_complete_20261022/CLEAN_READ_ME_081116.txt')
    house5_readme = readme.split('House 5')[1].split('House 6')[0]
    assert '7.Microwave' in house5_readme
    upstream_audit = read('docs/MICROWAVE_DATA_AUDIT_20261025.json')
    del upstream_audit
    prep = read('docs/MICROWAVE_MASK_PREPARATION_20261025.json')
    choice = read(f'{TCN}/source_selection.json')
    guard_source = read('src/virtual_submeter/data/protocol.py')
    guard_tests = read('tests/test_data_protocol.py')
    assert 'Source train, source validation and target households must be disjoint' in guard_source
    assert 'test_household_overlap_and_insufficient_isolation_are_rejected' in guard_tests
    rows, data, quality, frozen = [], {}, {}, []
    for h, (start, end, folder, ah, yh) in SOURCE.items():
        low, high = stamp(start), stamp(end)
        assert parts[h]['start'] == start and parts[h]['end'] == end
        ar, yr = read(f'{folder}/aggregate.csv', ah), read(f'{folder}/microwave.csv', yh)
        assert {r['household_id'] for r in ar+yr} == {h}
        t = np.array([stamp(r['timestamp']) for r in ar], dtype=np.int64)
        yt = np.array([stamp(r['timestamp']) for r in yr], dtype=np.int64)
        assert np.array_equal(t, yt) and np.array_equal(t, np.arange(low, high, 8))
        a = np.array([float(r['aggregate_power_w']) if r['aggregate_power_w'] else np.nan for r in ar])
        y = np.array([float(r['power_w']) if r['power_w'] else np.nan for r in yr])
        assert np.all(y[np.isfinite(y)] >= 0)
        mapping = parts[h]['origin']['channel_mapping']['microwave']
        assert mapping == catalog['households'][h.split('_')[1]]['microwave']
        assert {r['appliance_id'] for r in yr} == {'microwave:'+mapping[0]}
        ids, es = events(t, y)
        d = (t, a, y, ids, es)
        data[h] = d
        row, _, _ = summarize(h, 'original_14d', d, low, high)
        rows.append(row)
        flags = {}
        for r in yr:
            for f in json.loads(r['quality_flags']):
                flags[f] = flags.get(f, 0)+1
        quality[h] = {'mapping': mapping, 'label_flags': flags,
                      'label_missing_with_coverage_lt1': sum(not r['power_w'] and float(r['coverage']) < 1 for r in yr),
                      'label_missing_with_coverage_eq1': sum(not r['power_w'] and float(r['coverage']) == 1 for r in yr),
                      'aggregate_flags': sorted({f for r in ar for f in json.loads(r['quality_flags'])}),
                      'mapping_metadata_agrees': True}
        daily = []
        for day in np.unique(t//86400):
            dl, dh = max(low, int(day)*86400), min(high, (int(day)+1)*86400)
            r, _, _ = summarize(h, 'UTC_'+iso(dl)[:10], d, dl, dh)
            daily.append(r)
        assert sum(r['grid_points'] for r in daily) == len(t)
        assert sum(r['observed_on_centers'] for r in daily) == row['observed_on_centers']
        rows.extend(daily)
        if h == 'refit_5':
            for i in range(6):
                bl = stamp('2014-11-15T00:00:00Z') + i*2*86400
                r, strict, _ = summarize(h, f'block_{i}', d, bl, bl+2*86400)
                f = read(f'{FS}/validation/block_{i}.npz', keys=('t', 'y'))
                assert np.array_equal(f['t'], t[strict])
                np.testing.assert_allclose(f['y'], y[strict], rtol=1e-6, atol=1e-4)
                frozen.append(f)
                rows.append(r)
    totals = [r for r in rows if r['block'] == 'original_14d' and r['house'] != 'refit_5']
    strict_n = sum(r['strict']['windows'] for r in totals)
    partial_n = sum(r['partial']['windows'] for r in totals)
    partial_p = sum(r['partial']['positive_centers'] for r in totals)
    added_events = 0
    for h in ('refit_2', 'refit_19'):
        t,a,y,ids,es = data[h]
        sm, pm = masks(*data[h], int(t[0]), int(t[-1]+8))
        added_events += len(set(ids[pm & (y>=200)])-set(ids[sm & (y>=200)]))
    assert (strict_n, partial_n, partial_p, added_events) == (150069, 259806, 849, 51)
    assert prep['source_support']['partial_windows'] == partial_n
    # Fixed proposal boundaries, with a one-sided 1800s hold-out BEFORE validation begins.
    bl = stamp('2014-11-15T00:00:00Z')
    cut = stamp('2014-11-23T00:00:00Z')
    train_end = cut-1800
    proposal_support = []
    intervals = [(f'proposed_house5_train_block_{i}_audit_only',bl+i*2*86400,
                  bl+(i+1)*2*86400-(1800 if i==3 else 0)) for i in range(4)]
    intervals += [('retained_validation_block_4', cut, cut+2*86400),
                  ('retained_validation_block_5', cut+2*86400, cut+4*86400)]
    for name, lo, hi in intervals:
        r, _, _ = summarize('refit_5', name, data['refit_5'], lo, hi)
        # Conservative fully observed 127-point support; actual masked outer context can be larger.
        s127, _ = masks(*data['refit_5'], lo, hi, width=127)
        r['strict127_conservative'] = support(s127, data['refit_5'][2], data['refit_5'][3], data['refit_5'][4])
        proposal_support.append(r)
        rows.append(r)
    fpr = []
    for seed in (42,1019,2026):
        meta = read(f'{TCN}/models/TCN127/seed_{seed}/model.json')
        epoch = meta['selected']['epoch']
        for i, frozen_block in enumerate(frozen):
            score = read(f'{TCN}/models/TCN127/seed_{seed}/epoch_{epoch}_block_{i}.npz', keys=('t','scores'))
            assert np.array_equal(score['t'], frozen_block['t'])
            y = frozen_block['y'] >= 200
            s = score['scores']
            assert np.isfinite(s).all()
            # Highest threshold reaching at least .85 recall; ties are kept intact.
            threshold = float(np.sort(s[y])[len(s[y])-int(np.ceil(.85*y.sum()))])
            pred = s >= threshold
            fpr.append({'seed':seed,'epoch':epoch,'block':f'block_{i}',
                        'recall':float(pred[y].mean()),'fpr_at_recall85':float(pred[~y].mean()),
                        'threshold':threshold,'positive_centers':int(y.sum()),'negative_centers':int((~y).sum())})
    proposal = {
        'schema':'microwave_source_split_proposal_v1','disabled':True,
        'house5_training_authorized':False,'requires_human_approval':True,
        'current_roles_unchanged':True,'new_fits_this_audit':0,'suggested_future_maximum_fits':3,
        'experiment_kind_after_separate_approval_only':'same_household_temporal_engineering_experiment',
        'compatible_with_original_cross_household_protocol':False,
        'required_human_approved_exceptions':['House5 validation-only constraint','source_train/source_validation household disjointness'],
        'default_protocol_guard_must_remain_unchanged':True,
        'existing_freeze_protocol_entrypoint_compatible':False,
        'seeds':[42,1019,2026],'original_training_houses':['refit_2','refit_19'],
        'original_training_bounds':{h:{'start':v[0],'end':v[1]} for h,v in SOURCE.items() if h!='refit_5'},
        'house5_nominal_added_dates':{'start':iso(bl),'end':iso(cut),'frozen_blocks':[0,1,2,3]},
        'house5_allowed_training_interval_after_approval_only':{'start':iso(bl),'end':iso(train_end)},
        'house5_training_block_intervals_after_approval_only':[
            {'block':i,'start':iso(lo),'end':iso(hi)} for i,(_,lo,hi) in enumerate(intervals[:4])],
        'training_tail_safety_gap_seconds':1800,
        'retained_validation_blocks':[{'block':i,'start':iso(cut+(i-4)*2*86400),'end':iso(cut+(i-3)*2*86400)} for i in (4,5)],
        'discard_partial_edge_dates_and_unused_day':True,
        'house5_excluded_intervals':[
            {'start':SOURCE['refit_5'][0],'end':iso(bl)},
            {'start':iso(train_end),'end':iso(cut)},
            {'start':iso(cut+4*86400),'end':SOURCE['refit_5'][1]}],
        'windows_events_outer_context_must_stay_within_role_and_frozen_block':True,
        'max_outer_context_radius_seconds':504,'longer_context_requires_new_review':True,
        'missing_policy':'No filling; existing strict/partial contract unchanged',
        'validation_support':'Retain frozen strict C31 centers/labels; masked longer context must stop at each retained block boundary',
        'recipe':'WAIT_FOR_TASK8_SOURCE_ONLY_PRESELECTION; no development-based choice',
        'controls':'Reuse frozen old corresponding models and each epoch on identical retained block support; zero control retraining',
        'comparison_invariants':['same network','same updates','same batch size and positive/negative quotas','same original House2/19 normalization constants','same objective and p_ref','same seed','same checkpoint/threshold selection rule','same validation centers'],
        'importance_reference_prior':{'numerator':849,'denominator':259806,'value':849/259806,'fixed_from_original_training':True},
        'gates':'Keep original W50 identical-support development gates; all candidate models/selection/predictions sealed before any future House17 scoring',
        'final_houses_isolated':[8,9,20],'other_appliances_power_and_defaults_unchanged':True,
        'validation_is_fresh_independent':False,'date_movement_or_expansion_allowed':False,
    }
    summary = {
        'schema':'microwave_source_validation_audit_v1','fits':0,'data_roles_changed':False,
        'house17_data_read':False,'final_house_data_read':False,'house5_training_arrays_created':False,
        'input_policy':'Explicit allowlist; only original 14-day source CSVs, frozen House5 validation scores, source metadata. No model execution.',
        'source_support_verified':{'strict_windows':strict_n,'partial_windows':partial_n,'partial_positive_centers':partial_p,'new_supported_segments':added_events},
        'quality':quality,'house5_publisher_mapping_excerpt':house5_readme.strip(),
        'default_protocol_disjointness':{'guard':'src/virtual_submeter/data/protocol.py:26-31',
                                         'test':'tests/test_data_protocol.py:test_household_overlap_and_insufficient_isolation_are_rejected',
                                         'train_validation_target_groups_must_be_disjoint':True,
                                         'test_checks_train_target_overlap_and_short_isolation':True,
                                         'guard_covers_train_validation_overlap_as_well':True},
        'tcn_source_preselection':choice,'statistics':rows,'proposal_support':proposal_support,
        'house5_tcn127_fixed_recall_diagnostics':fpr,'proposal':proposal,
        'unknowns':['physical appliance identity not independently inspected','raw House5 resampling not independently replayed in this audit','upstream filled/zeroed samples unrecoverable','no fresh independent validation household','no controlled source training replay comparison computed','performance benefit of proposed training unknown'],
        'checks':{'six_frozen_block_time_and_labels_match':True,'original_mask_support_matches':True,
                  'daily_grid_and_on_count_closure':True,'power_bin_closure':True,'timestamp_eight_second_grid':True,
                  'no_non_allowlisted_inputs':True,'no_feature_arrays_or_normalization_or_sampler_created':True},
        'access_log':ACCESS,
    }
    out.mkdir(parents=True)
    write_json(out/'audit.json', summary)
    write_json(out/'proposal.disabled.json', proposal)
    flat = []
    for r in rows:
        f = {k:v for k,v in r.items() if not isinstance(v,dict)}
        for k,v in r.items():
            if isinstance(v,dict):
                f.update({f'{k}.{kk}':vv for kk,vv in v.items()})
        flat.append(f)
    with (out/'source_statistics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader();writer.writerows(flat)
    with (out/'fixed_recall_fpr.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(fpr[0]));writer.writeheader();writer.writerows(fpr)
    # Simple exportable, descriptive figure: same fixed power bins, normalized within house.
    palette=['#94a3b8','#64748b','#0f766e','#eab308','#dc2626']
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="280" viewBox="0 0 900 280">',
         '<rect width="900" height="280" fill="white"/><g font-family="Arial" font-size="15" fill="#172033">',
         '<text x="20" y="25">Observed microwave ON centers: fixed power bins (not independent events)</text>']
    for i,r in enumerate(r for r in rows if r['block']=='original_14d'):
        yy=50+i*48;xx=130
        svg.append(f'<text x="20" y="{yy+20}">{r["house"]}</text>')
        for j,n in enumerate(r['on_power_bins'].values()):
            width=650*n/r['observed_on_centers']
            svg.append(f'<rect x="{xx}" y="{yy}" width="{width}" height="30" fill="{palette[j]}"/>')
            xx+=width
    for j,label in enumerate(['200-600','600-1000','1000-1600','1600-2200','>2200 W']):
        xx=20+j*175
        svg.append(f'<rect x="{xx}" y="220" width="15" height="15" fill="{palette[j]}"/><text x="{xx+20}" y="233">{label}</text>')
    svg.append('<text x="20" y="268">200/600/1000/1600 enter the next bin; 2200 stays in 1600-2200. Source only; zero fits.</text></g></svg>')
    (out/'source_power_bins.svg').write_text('\n'.join(svg),encoding='utf-8')
    artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    write_json(out/'receipt.json',{'artifacts':artifact_hashes,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   'fits':0,'allowlisted_input_count':len(ACCESS),'checks_passed':summary['checks']})
    print(json.dumps({'output':str(out),'fits':0,'input_count':len(ACCESS),'source_support':summary['source_support_verified']},indent=2))


if __name__ == '__main__':
    main()
