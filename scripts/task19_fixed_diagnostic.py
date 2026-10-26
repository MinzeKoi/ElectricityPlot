"""Task19 fixed point diagnostic. No fitting, selection, or model mutation entry point."""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').is_file():
    ROOT = ROOT.parent
T18 = ROOT / 'outputs/accuracy/microwave-expanded-guard95-preparation-20261026-v3'
T10 = ROOT / 'outputs/accuracy/microwave-source5-temporal-20261025-v2'
T12 = ROOT / 'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
REG = '11dd4f8d9fdf68d2740c4b643a869ae462e21f4518eb309aacdb89eb92d11040'
SEEDS = (42, 1019, 2026)
FAMILIES = ('control', 'expanded')
SETS = ('train', 'block_4', 'block_5')
VIEWS = ('original_support', 'strict_core31')
CODE = ('virtual_submeter/__init__.py', 'virtual_submeter/models/__init__.py',
        'virtual_submeter/models/center_mask_tcn.py', 'virtual_submeter/models/arrays.py',
        'virtual_submeter/data/__init__.py', 'virtual_submeter/data/common.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(a):
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + b'\0' + str(a.shape).encode() + b'\0')
    h.update(a.tobytes())
    return h.hexdigest()


def fresh(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def model_path(family, seed):
    return (T12 / 'models/unit_BCE' if family == 'control' else T18 / 'models/expanded') / f'seed_{seed}'


def data_path(name):
    return T18 / ('train.npz' if name == 'train' else f'validation/{name}.npz')


def allowed_inputs():
    paths = [T18 / n for n in ('registration.json', 'execution_delivery_manifest.json',
             'source_selection.json', 'protocol.md', 'normalization.json')]
    paths += [data_path(n) for n in SETS] + [T10 / 'train.npz']
    paths += [model_path(f, s) / n for f in FAMILIES for s in SEEDS for n in ('model.json', 'weights.pt')]
    paths += [T18 / 'source' / n for n in CODE]
    return paths


def snapshot(paths):
    return {str(p.resolve()): dict(sha256=sha(p), mtime_ns=p.stat().st_mtime_ns, size=p.stat().st_size) for p in paths}


def check_open(path, mode, flags, run, inputs, readonly=False):
    if isinstance(path, int):
        return
    p = Path(os.fsdecode(path)).resolve()
    writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    if p.is_relative_to(run):
        if readonly and writing:
            raise PermissionError('Task19 verify/report are read-only')
        return
    if writing:
        raise PermissionError('Task19 writes only inside its fresh output directory')
    runtime_root = Path(sys.prefix).resolve()
    if p.is_relative_to(ROOT) and p not in inputs and not p.is_relative_to(runtime_root):
        raise PermissionError('Task19 read outside exact allowed paths: ' + str(p))
    # Runtime/library reads outside workspace are needed by PyTorch; no external data routing.


def isolation(run, readonly=False):
    inputs = {p.resolve() for p in allowed_inputs()}
    def audit(event, args):
        if event == 'open':
            check_open(args[0], args[1], args[2], run.resolve(), inputs, readonly)
        elif event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
            raise PermissionError('Task19 network/subprocess disabled')
    sys.addaudithook(audit)
    def forbidden(*args, **kwargs):
        raise RuntimeError('Task19 prohibits backward and optimizer construction')
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.autograd.grad = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    torch.set_grad_enabled(False)


def runtime():
    sys.path.insert(0, str(T18 / 'source'))
    from virtual_submeter.models.center_mask_tcn import load_state_model, predict_windows, tensor_input
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    return load_state_model, predict_windows, tensor_input


def validate_dataset(d, name):
    n = len(d['y'])
    require(d['x'].shape == (n, 127) and d['mask'].shape == (n, 127), 'Window shape differs')
    require(np.array_equal(np.isfinite(d['x']), d['mask']), 'Mask mismatch')
    require(not np.isinf(d['x']).any() and np.all(d['x'][d['mask']] >= 0), 'Invalid power input')
    require(np.isfinite(d['y']).all() and np.all(d['y'] >= 0), 'Invalid label')
    require(set(np.unique(d['house'])).issubset({0, 1, 2}), 'Unexpected house')
    core = d['mask'][:, 48:79]
    require(np.all(core[:, 15]) and np.all(core.sum(1) >= 24), 'Training support invalid')
    require(not np.any((~core[:, :-1]) & (~core[:, 1:])), 'Consecutive missing core')
    if name != 'train':
        require(core.all() and np.all(d['house'] == 2), 'Validation must be strict House5')
        require(np.all(d['block'] == int(name[-1])), 'Validation block mapping differs')
        d['source_role'] = np.full(n, 'house5_validation_' + name, dtype='U32')
    keys = np.rec.fromarrays([d['house'], d['source_role'], d['t']], names='house,role,t')
    require(len(np.unique(keys)) == n, 'Duplicate window ID')
    require(np.all(d['t'] % 8 == 0), 'Grid alignment differs')
    return core.all(1)


def load_dataset(name):
    with np.load(data_path(name), allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    strict = validate_dataset(d, name)
    expected = {'train': (615922, 2994), 'block_4': (19127, 136), 'block_5': (15224, 65)}
    require((len(d['y']), int((d['y'] >= 200).sum())) == expected[name], 'Frozen support differs')
    return d, strict


def identities(d, strict):
    return {**{k: d[k] for k in ('t', 'house', 'source_role', 'event', 'raw_index')},
            'positive': d['y'] >= 200, 'strict_core31': strict, 'row_index': np.arange(len(strict))}


def infer(meta, model, d, strict, ordinary, tensor):
    """Original ordinary interface for strict rows; identical network for partial rows."""
    require(not model.training and not torch.is_grad_enabled(), 'Inference isolation absent')
    out = np.empty(len(strict), np.float32)
    for is_strict in (True, False):
        ids = np.flatnonzero(strict == is_strict)
        for start in range(0, len(ids), 8192):
            ix = ids[start:start+8192]
            x = d['x'][ix]
            if is_strict:
                p = ordinary(meta, model, x)
            else:
                z = tensor(meta, x, d['mask'][ix])
                model.eval()
                with torch.inference_mode():
                    p = np.concatenate([torch.sigmoid(model(b)).numpy() for b in z.split(meta['batch_size'])])
            out[ix] = p
    require(np.isfinite(out).all() and np.all((out >= 0) & (out <= 1)), 'Invalid prediction')
    return out


def ordinary_readback_batches(ids, dataset):
    if dataset == 'train':
        return (ids[:256], ids[((len(ids)-1)//256)*256:])
    return (ids,)


def ap_grouped(y, p):
    """Noninterpolated AP = sum(group positive fraction * precision at tied group end)."""
    total = int(y.sum())
    if not total:
        return None
    order = np.argsort(-p, kind='stable')
    q, z = p[order], y[order]
    end = np.r_[np.flatnonzero(q[:-1] != q[1:]), len(q)-1]
    tp = np.cumsum(z, dtype=np.int64)[end]
    added = np.diff(np.r_[0, tp])
    return float(np.sum((added / total) * (tp / (end + 1))))


def ratio(a, b):
    return float(a / b) if b else None


def metrics(y, p, threshold):
    require(y.dtype == np.bool_ and y.shape == p.shape and np.isfinite(p).all(), 'Invalid metric arrays')
    pred = p >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    ap = ap_grouped(y, p)
    return dict(n=len(y), positives=int(y.sum()), negatives=int((~y).sum()),
                prevalence=ratio(int(y.sum()), len(y)), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn), f1=ratio(2*tp, 2*tp+fp+fn),
                fpr=ratio(fp, fp+tn), fp_per_1000_off=ratio(1000*fp, fp+tn),
                ap=ap, ap_status='defined' if ap is not None else 'no_positive_support',
                recall_status='defined' if tp+fn else 'no_positive_support',
                threshold=threshold)


def group_specs(d, dataset):
    """All memberships fixed by data identity, never by model outcomes."""
    h, r, t, e = (d[k] for k in ('house', 'source_role', 't', 'event'))
    n = len(h)
    yield 'dataset', dataset, np.arange(n)
    if dataset == 'train':
        yield 'aggregate', 'original_train', np.flatnonzero(np.arange(n) < 336704)
        yield 'aggregate', 'added_dates', np.flatnonzero(np.arange(n) >= 336704)
        for code, house in enumerate((2, 19, 5)):
            yield 'partition', f'original_H{house}', np.flatnonzero((h == code) & (np.arange(n) < 336704))
        for code, house in enumerate((2, 19)):
            yield 'partition', f'added_H{house}', np.flatnonzero((h == code) & (np.arange(n) >= 336704))
    else:
        yield 'partition', dataset, np.arange(n)
    # Lexicographic grouping does not depend on y or predictions.
    for level, arrays in (('role', [h, r]), ('date', [h, r, t//86400]),
                          ('event_date', [h, r, t//86400, e])):
        key = np.rec.fromarrays(arrays, names=','.join('k'+str(i) for i in range(len(arrays))))
        order = np.argsort(key, kind='stable')
        ends = np.r_[0, np.flatnonzero(key[order][1:] != key[order][:-1])+1, n]
        for a, b in zip(ends[:-1], ends[1:]):
            ix = order[a:b]
            values = [str(v[ix[0]]) for v in arrays]
            values[0] = 'H' + str((2, 19, 5)[int(h[ix[0]])])
            if level in ('date', 'event_date'):
                values[2] = datetime.fromtimestamp(int(t[ix[0]]//86400)*86400, timezone.utc).strftime('%Y-%m-%d')
            yield level, '|'.join(values), ix


def support(d, ix):
    y = d['positive'][ix]
    order = np.lexsort((d['t'][ix], d['source_role'][ix], d['house'][ix]))
    j = ix[order]
    pos = d['positive'][j]
    boundary = np.r_[True, (np.diff(d['t'][j]) != 8) | (d['house'][j][1:] != d['house'][j][:-1]) |
                       (d['source_role'][j][1:] != d['source_role'][j][:-1])]
    starts = pos & (boundary | np.r_[True, ~pos[:-1]])
    events = set(zip(d['house'][ix][y].tolist(), d['source_role'][ix][y].tolist(), d['event'][ix][y].tolist()))
    return dict(observable_positive_fragments=int(starts.sum()), supported_event_ids=len(events),
                first_utc=int(d['t'][ix].min()) if len(ix) else None,
                last_utc=int(d['t'][ix].max()) if len(ix) else None)


def rows_for(d, p, threshold, family, seed, dataset):
    rows = []
    for level, group, original in group_specs(d, dataset):
        for view in VIEWS:
            ix = original if view == 'original_support' else original[d['strict_core31'][original]]
            row = dict(family=family, seed=seed, dataset=dataset, level=level, group=group, view=view)
            row.update(metrics(d['positive'][ix], p[ix], threshold), **support(d, ix))
            rows.append(row)
    return rows


def config():
    return dict(task='Task19_fixed_point_fit_generalization_diagnostic', registration_sha256=REG,
                families=list(FAMILIES), seeds=list(SEEDS), datasets=list(SETS), views=list(VIEWS),
                allowed_paths=[str(p.resolve()) for p in allowed_inputs()],
                runtime_read_root=str(Path(sys.prefix).resolve()),
                authorization=dict(source_chat='01a16d7b-478e-7780-a999-88abe71e8c94',
                    source_call='call_cd2c248bdc154827ba1069b2e2faf96f',
                    human_quote='允许新开并执行拟合／泛化诊断'),
                new_fits=0, optimizer_updates=0, threshold_searches=0, checkpoint_searches=0,
                development_reads=0, final_reads=0, default_changes=0,
                label='center original y >= 200W', threshold='model.json decision_threshold_probability unchanged; >=',
                ap='noninterpolated step AP; all equal scores grouped; no positive support => null',
                groups='all dataset/partition/role/UTC date/UTC date-event groups; data-defined before inference',
                event_warning='observable threshold fragments and supported IDs; not independent physical cycles',
                strict_view='original mask[:,48:79].all; unchanged C127 input and mask',
                partial_inference='frozen tensor_input + eval + inference_mode, same normalization; ordinary API rejects partial core',
                forbidden_data='House17 development and final Houses8/9/20; no values/hash/shape',
                external_validity='same-house engineering contrast; added dates already audited and used by expanded; no new blind test')


def prepare(run):
    require(not run.exists(), 'Task19 requires a completely new directory')
    run.mkdir(parents=True)
    fresh(run / 'config.json', config())
    shutil.copyfile(Path(__file__), run / 'diagnostic.py')
    fresh(run / 'pre_registration.json', dict(config_sha256=sha(run/'config.json'), source_sha256=sha(run/'diagnostic.py'),
          thresholds={f'{f}_{s}': read(model_path(f, s)/'model.json')['decision_threshold_probability'] for f in FAMILIES for s in SEEDS}))
    print('Prepared', run, flush=True)


def execute(run):
    require(not (run/'started.json').exists(), 'No repeated execution/overwrite')
    require(read(run/'config.json') == config(), 'Frozen config differs')
    pre = read(run/'pre_registration.json')
    require(sha(run/'config.json') == pre['config_sha256'] and sha(run/'diagnostic.py') == pre['source_sha256'], 'Preregistered code/config changed')
    fresh(run/'started.json', dict(unix=time.time(), new_fits=0, optimizer_updates=0))
    isolation(run)
    before = snapshot(allowed_inputs())
    require(sha(T18/'registration.json') == REG, 'Task18 registration identity differs')
    manifest = read(T18/'execution_delivery_manifest.json')
    require(len(manifest['artifacts']) == 289 and manifest['registration_sha256'] == REG and manifest['development_evaluated'] is False, 'Task18 manifest header differs')
    require(read(T18/'source_selection.json')['expanded_eligible'] is False, 'Expected source rejected expanded family')
    # Bind only authorized files. Do not read other epochs/checkpoints or raw/development/final files.
    for path in allowed_inputs():
        if path.is_relative_to(T18) and path.name != 'execution_delivery_manifest.json':
            key = path.relative_to(T18).as_posix()
            require(manifest['artifacts'][key] == before[str(path.resolve())]['sha256'], 'Manifest mismatch: '+key)
    fresh(run/'inputs_before.json', before)
    load, ordinary, tensor = runtime()
    all_rows, model_info, data_info = [], {}, {}
    (run/'identities').mkdir()
    (run/'predictions').mkdir()
    for dataset in SETS:
        d, strict = load_dataset(dataset)
        if dataset == 'train':
            with np.load(T10/'train.npz', allow_pickle=False) as old:
                for k in old.files:
                    a = old[k]
                    require(d[k].dtype == a.dtype and d[k][:len(a)].tobytes() == a.tobytes(), 'Original prefix changed: '+k)
            require(np.all(d['source_role'][336704:] == 'extended14d'), 'Added role mapping changed')
            for code, start, stop, count, positives in ((0, '2014-10-14', '2014-10-28', 144557, 404), (1, '2014-10-19', '2014-11-02', 134661, 315)):
                ix = (d['house'] == code) & (np.arange(len(strict)) >= 336704)
                lo, hi = (int(datetime.fromisoformat(v).replace(tzinfo=timezone.utc).timestamp()) for v in (start, stop))
                require(int(ix.sum()) == count and int((d['y'][ix]>=200).sum()) == positives and np.all((d['t'][ix]>=lo)&(d['t'][ix]<hi)), 'Additional date mapping differs')
        ident = identities(d, strict)
        np.savez_compressed(run/f'identities/{dataset}.npz', **ident)
        data_info[dataset] = dict(input_path=str(data_path(dataset)), file_sha256=sha(data_path(dataset)),
                                  fields={k: array_sha(v) for k,v in d.items()}, identity_sha256=sha(run/f'identities/{dataset}.npz'),
                                  n=len(strict), positives=int(ident['positive'].sum()), strict_n=int(strict.sum()), strict_positive=int(ident['positive'][strict].sum()))
        for family in FAMILIES:
            for seed in SEEDS:
                key = f'{family}_{seed}'
                meta, model = load(model_path(family, seed))
                threshold = meta['decision_threshold_probability']
                require(threshold == pre['thresholds'][key], 'Frozen threshold changed')
                require(meta['input_log_mean'] == 5.370561599731445 and meta['input_log_std'] == .8337093591690063, 'Normalization changed')
                state_before = {k: array_sha(v.detach().numpy()) for k,v in model.state_dict().items()}
                start = time.monotonic()
                p = infer(meta, model, d, strict, ordinary, tensor)
                require(state_before == {k: array_sha(v.detach().numpy()) for k,v in model.state_dict().items()}, 'Model weights mutated')
                require(all(q.grad is None for q in model.parameters()), 'Unexpected parameter gradients')
                binding = dict(family=family, seed=seed, dataset=dataset, model_json_sha256=sha(model_path(family,seed)/'model.json'),
                     weights_sha256=sha(model_path(family,seed)/'weights.pt'), input_sha256=data_info[dataset]['file_sha256'],
                     identity_sha256=data_info[dataset]['identity_sha256'], threshold=threshold)
                dest = run/f'predictions/{key}_{dataset}.npz'
                np.savez_compressed(dest, probability=p, row_index=ident['row_index'], binding_json=np.array(json.dumps(binding, sort_keys=True)))
                fresh(dest.with_suffix('.json'), binding | dict(prediction_sha256=sha(dest), probability_sha256=array_sha(p)))
                all_rows.extend(rows_for(ident, p, threshold, family, seed, dataset))
                model_info[key] = binding | dict(selected_epoch=meta['selected']['epoch'], original_updates=meta['updates'], parameter_count=meta['parameter_count'])
                print(f'{key} {dataset}: {len(p)} complete, {time.monotonic()-start:.1f}s', flush=True)
                del model
        del d
    # Pooled validation support is predeclared, never a substituted block score.
    val = [load_identity(run, n) for n in SETS[1:]]
    joined = {k: np.concatenate([v[k] for v in val]) for k in val[0]}
    for family in FAMILIES:
        for seed in SEEDS:
            p = np.concatenate([load_prediction(run, family, seed, n) for n in SETS[1:]])
            for view in VIEWS:
                all_rows.append(dict(family=family, seed=seed, dataset='validation_pooled', level='aggregate', group='validation_pooled', view=view,
                                     **metrics(joined['positive'], p, pre['thresholds'][f'{family}_{seed}']), **support(joined, np.arange(len(p)))))
    fresh(run/'data_identity.json', data_info)
    fresh(run/'model_identity.json', model_info)
    fresh(run/'metrics.json', all_rows)
    with (run/'metrics.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)
    after = snapshot(allowed_inputs())
    require(before == after, 'Inputs changed during inference')
    fresh(run/'inputs_after.json', after)
    fresh(run/'inference_complete.json', dict(prediction_files=18, model_count=6, metrics_rows=len(all_rows), new_fits=0, optimizer_updates=0,
                 source_inputs_hash_mtime_unchanged=True, access_policy='exact input allowlist; output-only writes; network/subprocess/backward/optimizer blocked'))
    print('Inference complete', len(all_rows), 'metric rows', flush=True)


def load_identity(run, dataset):
    with np.load(run/f'identities/{dataset}.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_prediction(run, family, seed, dataset):
    with np.load(run/f'predictions/{family}_{seed}_{dataset}.npz', allow_pickle=False) as z:
        require(set(z.files) == {'probability', 'row_index', 'binding_json'}, 'Unexpected cache fields')
        p = z['probability']
        require(np.array_equal(z['row_index'], np.arange(len(p))), 'Prediction mapping differs')
        binding = json.loads(str(z['binding_json']))
        require((binding['family'], binding['seed'], binding['dataset']) == (family, seed, dataset), 'Cache identity mismatch')
        meta = read(model_path(family, seed)/'model.json')
        require(binding['threshold'] == meta['decision_threshold_probability'], 'Cache threshold differs')
        require(binding['model_json_sha256'] == sha(model_path(family,seed)/'model.json') and binding['weights_sha256'] == sha(model_path(family,seed)/'weights.pt'), 'Model identity mismatch')
        require(binding['input_sha256'] == sha(data_path(dataset)) and binding['identity_sha256'] == sha(run/f'identities/{dataset}.npz'), 'Data identity mismatch')
        side = read(run/f'predictions/{family}_{seed}_{dataset}.json')
        require(side == binding | dict(prediction_sha256=sha(run/f'predictions/{family}_{seed}_{dataset}.npz'), probability_sha256=array_sha(p)), 'Cache sidecar mismatch')
    require(p.ndim == 1 and p.dtype == np.float32 and np.isfinite(p).all() and np.all((p>=0)&(p<=1)), 'Invalid cached probabilities')
    return p


def independent_ap(y, p):
    """Separate ascending unique/bincount implementation, including equal-score groups."""
    positives = sum(bool(v) for v in y)
    if positives == 0:
        return None
    values, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    positive_counts = np.bincount(inverse, weights=y.astype(np.int64), minlength=len(values))
    cumulative_tp = np.cumsum(positive_counts[::-1])
    cumulative_n = np.cumsum(counts[::-1])
    return float(sum(float(k)*float(a)/float(b) for k,a,b in zip(positive_counts[::-1], cumulative_tp, cumulative_n)) / positives)


def independent_metrics(y, p, threshold):
    tp = fp = fn = tn = 0
    for truth, prob in zip(y, p, strict=True):
        if bool(truth):
            if float(prob) >= threshold: tp += 1
            else: fn += 1
        elif float(prob) >= threshold: fp += 1
        else: tn += 1
    def r(a,b): return float(a/b) if b else None
    ap = independent_ap(y,p)
    return dict(n=tp+fp+fn+tn, positives=tp+fn, negatives=fp+tn, prevalence=r(tp+fn,tp+fp+fn+tn),
                tp=tp,fp=fp,fn=fn,tn=tn,precision=r(tp,tp+fp),recall=r(tp,tp+fn),f1=r(2*tp,2*tp+fp+fn),
                fpr=r(fp,fp+tn),fp_per_1000_off=r(1000*fp,fp+tn),ap=ap,
                ap_status='defined' if ap is not None else 'no_positive_support',
                recall_status='defined' if tp+fn else 'no_positive_support',threshold=threshold)


def matrix_check(run):
    expected = {f'{f}_{s}_{d}.npz' for f in FAMILIES for s in SEEDS for d in SETS}
    actual = {p.name for p in (run/'predictions').glob('*.npz')}
    require(expected == actual, 'Incomplete/extra prediction family-seed-block matrix')


def verify_delivery_manifest(run):
    path = run/'delivery_manifest.json'
    if not path.exists():
        return
    manifest = read(path)
    require(manifest['status']=='complete_fixed_diagnostic', 'Delivery status differs')
    expected = {str((run / n).resolve()): v for n,v in manifest['artifacts'].items()}
    require(all(Path(p).is_relative_to(run.resolve()) for p in expected), 'Manifest path escape')
    actual_files = {str(p.resolve()) for p in run.rglob('*') if p.is_file() and p != path}
    require(actual_files == set(expected), 'Delivery artifact matrix changed')
    require(snapshot([Path(p) for p in expected]) == expected, 'Sealed delivery hash/mtime changed')


def compare_metrics(actual, expected):
    for k, v in expected.items():
        if isinstance(v, float):
            require(actual[k] is not None and abs(actual[k]-v) <= 2e-12, 'Independent metric differs: '+k)
        else:
            require(actual[k] == v, 'Independent metric differs: '+k)


def verify(run, ordinary_check=True):
    """Read-only: no writes, imports cannot create bytecode, no report mutation."""
    require(read(run/'config.json') == config(), 'Config changed')
    pre = read(run/'pre_registration.json')
    require(sha(run/'config.json') == pre['config_sha256'] and sha(run/'diagnostic.py') == pre['source_sha256'], 'Code/config changed')
    require(read(run/'inputs_before.json') == snapshot(allowed_inputs()) == read(run/'inputs_after.json'), 'Source hash/mtime changed')
    matrix_check(run)
    verify_delivery_manifest(run)
    info = read(run/'model_identity.json')
    require(set(info)=={f'{f}_{s}' for f in FAMILIES for s in SEEDS}, 'Model identity matrix differs')
    for f in FAMILIES:
        for s in SEEDS:
            meta=read(model_path(f,s)/'model.json');item=info[f'{f}_{s}']
            for k,v in dict(threshold=meta['decision_threshold_probability'], selected_epoch=meta['selected']['epoch'],
                            original_updates=meta['updates'], parameter_count=meta['parameter_count'],
                            model_json_sha256=sha(model_path(f,s)/'model.json'),weights_sha256=sha(model_path(f,s)/'weights.pt')).items():
                require(item[k]==v,'Report model metadata differs: '+k)
    stored = read(run/'metrics.json')
    key_fields = ('family','seed','dataset','level','group','view')
    indexed = {tuple(row[k] for k in key_fields): row for row in stored}
    require(len(indexed) == len(stored), 'Duplicate metric rows')
    seen, checks = set(), 0
    load, ordinary, tensor = runtime()
    loaded = {}
    for dataset in SETS:
        d, strict = load_dataset(dataset)
        ident = load_identity(run, dataset)
        expected = identities(d, strict)
        require(set(expected) == set(ident) and all(np.array_equal(expected[k],ident[k]) for k in expected), 'Persisted identity mapping differs')
        require(read(run/'data_identity.json')[dataset]['fields'] == {k:array_sha(v) for k,v in d.items()}, 'Original fields changed')
        loaded[dataset] = ident
        for family in FAMILIES:
            for seed in SEEDS:
                p = load_prediction(run, family, seed, dataset)
                require(len(p) == len(ident['positive']), 'Cache length differs')
                th = pre['thresholds'][f'{family}_{seed}']
                if ordinary_check:
                    meta, model = load(model_path(family,seed))
                    # Preserve original 256-row batch boundaries; changing the final
                    # batch length can change CPU convolution rounding by a few ulps.
                    ids = np.flatnonzero(strict)
                    for batch in ordinary_readback_batches(ids, dataset):
                        q = ordinary(meta, model, d['x'][batch])
                        require(np.array_equal(q, p[batch]), 'Ordinary predict readback not exact')
                    partial = np.flatnonzero(~strict)[:256]
                    if len(partial):
                        with torch.inference_mode():
                            q = torch.sigmoid(model(tensor(meta,d['x'][partial],d['mask'][partial]))).numpy()
                        require(np.array_equal(q,p[partial]), 'Partial forward readback not exact')
                    del model
                for level, group, base in group_specs(ident,dataset):
                    for view in VIEWS:
                        ix = base if view == VIEWS[0] else base[ident['strict_core31'][base]]
                        key = (family,seed,dataset,level,group,view)
                        require(key in indexed, 'Missing predefined group')
                        compare_metrics(indexed[key],independent_metrics(ident['positive'][ix],p[ix],th))
                        compare_metrics(indexed[key],support(ident,ix))
                        seen.add(key); checks += 1
        del d
    joined = {k:np.concatenate([loaded[n][k] for n in SETS[1:]]) for k in loaded['block_4']}
    for f in FAMILIES:
        for s in SEEDS:
            p = np.concatenate([load_prediction(run,f,s,n) for n in SETS[1:]])
            for view in VIEWS:
                key=(f,s,'validation_pooled','aggregate','validation_pooled',view)
                require(key in indexed, 'Missing pooled validation')
                compare_metrics(indexed[key],independent_metrics(joined['positive'],p,pre['thresholds'][f'{f}_{s}']))
                compare_metrics(indexed[key],support(joined,np.arange(len(p))))
                seen.add(key);checks+=1
    require(seen == set(indexed), 'Unexpected/missing groups')
    # Ensure CSV and JSON expose the same records, including nulls as empty CSV cells.
    with (run/'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows=list(csv.DictReader(stream))
    require(len(csv_rows)==len(stored), 'CSV rows differ')
    for a,b in zip(csv_rows,stored,strict=True):
        require(all(a[k] == ('' if v is None else str(v)) for k,v in b.items()), 'CSV/JSON mismatch')
    return dict(status='verified', model_count=6, prediction_caches=18, independent_group_checks=checks,
                scalar_confusion_exact=True, grouped_ap_independent_max_tolerance=2e-12,
                ordinary_readback='all strict validation + first 256 and original final <=256 strict train batch; exact',
                partial_readback='fixed first 256 partial train; exact',
                input_hash_mtime_unchanged=True, new_fits=0, optimizer_updates=0)


def report_text(run):
    matrix_check(run)
    verify_delivery_manifest(run)
    rows = read(run/'metrics.json')
    def get(f,s,g,v):
        level = 'dataset' if g=='train' else ('aggregate' if g in ('original_train','validation_pooled') else 'partition')
        matches=[r for r in rows if r['family']==f and r['seed']==s and r['group']==g and r['view']==v and r['level']==level]
        require(len(matches)==1, 'Ambiguous report cell')
        return matches[0]
    def fmt(v): return '无定义' if v is None else f'{v:.4f}'
    lines=['# Task19：固定点分类拟合／泛化诊断', '',
      '本报告为六个既有冻结模型的固定点分类诊断。不是正式 L95/H95 事件门控验收，不能用于模型晋升、默认切换或宣称功率改善。新增训练0、优化更新0、阈值搜索0、checkpoint搜索0；未读取 House17 或最终 Houses8/9/20 的数值、形状或hash。', '',
      'control为Task12 unit_BCE三种子；expanded为Task18来源资格已拒绝的三种子。唯一分类点是各自model.json冻结的原source85阈值，以概率≥阈值为开。所有窗口逐一推理，不重新采样。expanded在新增2/19日期上已经训练，其成绩只表示拟合。control在这些日期没有训练，但标签已经审计并用于expanded，不能称为新独立盲测。House5两个suffix块曾用于选模和定阈值，也不是独立测试集。', '',
      '严格视图只按原mask核心31点全观测过滤，127上下文及掩码完全保留。全部支持还包含训练允许的部分缺失核心；两视图分开分析。AP为按同分组尾精度加权的非插值step AP，不是梯形PR曲线面积，也不是另选决策阈值。零正例分组AP/Recall为null并带状态。片段计数是可观察真值阈值片段，不等于独立物理cycle。', '',
      '## 每种子固定阈值', '', '|family|seed|冻结阈值|选定epoch|原训练更新数|', '|---|---:|---:|---:|---:|']
    model=read(run/'model_identity.json')
    for f in FAMILIES:
        for s in SEEDS:
            m=model[f'{f}_{s}']; lines.append(f'|{f}|{s}|{m["threshold"]:.10f}|{m["selected_epoch"]}|{m["original_updates"]}|')
    groups=('original_train','train','original_H2','original_H19','original_H5','added_H2','added_H19','block_4','block_5','validation_pooled')
    for view in VIEWS:
        lines += ['', f'## {view}：每种子数值', '', '|family|seed|分组|n|正例/开机率|片段|P|R|F1|AP|FP/千关闭点|TP/FP/FN/TN|', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for f in FAMILIES:
            for s in SEEDS:
                for g in groups:
                    r=get(f,s,g,view)
                    lines.append(f'|{f}|{s}|{g}|{r["n"]}|{r["positives"]}/{r["prevalence"]:.5%}|{r["observable_positive_fragments"]}|{fmt(r["precision"])}|{fmt(r["recall"])}|{fmt(r["f1"])}|{fmt(r["ap"])}|{fmt(r["fp_per_1000_off"])}|{r["tp"]}/{r["fp"]}/{r["fn"]}/{r["tn"]}|')
    lines += ['', '## 证据与判断', '',
      '以下汇总为三种子算术均值，先看共同严格支持；每种子及不同开机率的原始计数见上表和完整CSV。', '',
      '|family|分组|P均值|R均值|F1均值|AP均值|FP/千关闭点均值|', '|---|---|---:|---:|---:|---:|---:|']
    mean={}
    for f in FAMILIES:
        for g in groups:
            rr=[get(f,s,g,'strict_core31') for s in SEEDS]
            mean[f,g]={k:float(np.mean([r[k] for r in rr])) for k in ('precision','recall','f1','ap','fp_per_1000_off')}
            m=mean[f,g];lines.append('|'+f+'|'+g+'|'+'|'.join(fmt(m[k]) for k in m)+'|')
    lines += ['', '数值解释（不据此改阈值或选模型）：', '']
    for f in FAMILIES:
        a,b=mean[f,'original_H5'],mean[f,'validation_pooled']
        lines.append(f'- {f}：House5前缀→suffix，P {a["precision"]:.4f}→{b["precision"]:.4f}，R {a["recall"]:.4f}→{b["recall"]:.4f}，F1 {a["f1"]:.4f}→{b["f1"]:.4f}，AP {a["ap"]:.4f}→{b["ap"]:.4f}，FP/千关闭点 {a["fp_per_1000_off"]:.2f}→{b["fp_per_1000_off"]:.2f}。')
    for h in (2,19):
        a,b=mean['control',f'original_H{h}'],mean['control',f'added_H{h}']
        lines.append(f'- control House{h}原训练→未训练新增日期：P {a["precision"]:.4f}→{b["precision"]:.4f}，R {a["recall"]:.4f}→{b["recall"]:.4f}，F1 {a["f1"]:.4f}→{b["f1"]:.4f}，AP {a["ap"]:.4f}→{b["ap"]:.4f}，FP/千关闭点 {a["fp_per_1000_off"]:.2f}→{b["fp_per_1000_off"]:.2f}。这是一项冻结模型跨日期工程对照。')
    a,b=mean['control','original_train'],mean['expanded','original_train']
    c,d=mean['control','validation_pooled'],mean['expanded','validation_pooled']
    lines.append(f'- 相同原训练支持control→expanded：F1 {a["f1"]:.4f}→{b["f1"]:.4f}、AP {a["ap"]:.4f}→{b["ap"]:.4f}；相同suffix：F1 {c["f1"]:.4f}→{d["f1"]:.4f}、AP {c["ap"]:.4f}→{d["ap"]:.4f}。数据量与更新数同时改变，且冻结阈值和选定epoch不同，不能把差异归因于某个单因素。')
    for f,training in (('control','original_train'),('expanded','train')):
        rr=[get(f,s,training,'original_support') for s in SEEDS]
        lines.append(f'- {f}自己的完整来源训练分布（含原部分缺失支持）：P均值 {np.mean([r["precision"] for r in rr]):.4f}、R均值 {np.mean([r["recall"] for r in rr]):.4f}、F1均值 {np.mean([r["f1"] for r in rr]):.4f}、AP均值 {np.mean([r["ap"] for r in rr]):.4f}。这里control只使用original_train，expanded使用完整train；control在扩展train的总分混入未训练日期，不能冒充其训练拟合分数。')
    lines += ['', 'House5前缀到suffix并非所有指标同时下降：两个family的Recall均升、FPR均降，而开机率明显降低。P/F1/AP下降存在，但不能称为误报率恶化或全面泛化崩溃。旧control在House2/19新增日期上则同时出现Recall下降和FPR上升，提供比单独AP下降更直接的同户跨日期性能退化证据；日期输入/标签条件改变仍可能参与解释。', '',
      'expanded相对control没有稳定的“训练与验证同时改善”：相同原训练支持平均F1略升但AP下降，suffix平均F1下降而AP略升；新增2/19日期的拟合指标改善不能当作泛化收益。这些结果更支持保留原来源拒绝结论，而非晋升expanded。']
    lines += ['', 'Precision满足 P=R×开机率/[R×开机率+FPR×(1−开机率)]。因此罕见开机时，即使Recall很高、FPR只有百分之一量级，也会出现大量FP和低Precision。上表同时给出开机率、FPR、AP及混淆数，Precision/AP的单独下降不足以证明跨日期排序能力退化；AP也受开机率变化影响。固定阈值下的低分不能单独证明概率校准差，本诊断没有实施重新校准。', '',
      '完整训练支持的低P/F1及残余FP/FN说明，模型并未在大训练分布上达到接近完美的点分类拟合。小集合曾达到100%只说明那一小集合可被记忆，不证明大分布上的表达能力、优化、输入可辨识性都足够。这里无法唯一分离优化不足、表达不足、输入歧义或标签噪声。', '',
      '## 局限与下一步', '',
      '所有日期和事件分组均按角色/UTC日期/已有事件ID预定义，无按成绩挑日期。按日期的片段可能被午夜或观测缺口切断；窗口高度相关，三种子也不是三份独立数据，不据此作独立cycle显著性推断。验证块曾参与checkpoint与阈值选择，完整训练中的部分缺失视图不能直接与严格验证做因果比较。跨日期差异也可能由背景负载、类比例、输入支持和标签改变造成。', '',
      '下一轮唯一优先方向：预先登记同户按时间分块的困难负例辨识诊断，检查冻结模型FP窗口与真实开机窗口在相同核心完整支持、相近aggregate上下文下是否可区分。可证伪预期是：若错误主要来自可分但未学好的背景模式，预先定义的负载特征应能在留出日期保持分离；若匹配输入仍频繁对应冲突标签，则应优先复核输入可辨识性/标签，而不能继续盲目增加训练。本轮不执行新训练、不重新选模型或阈值。', '',
      '## 交付与复核', '',
      'metrics.csv / metrics.json保存每family×seed×视图×数据分组/角色/UTC日期/日期事件的全部P/R/F1/AP/FPR、混淆和支持。predictions/含18份原始float32完整概率缓存（6份训练、12份验证），每份绑定模型、阈值、输入及共享身份文件hash；identities/仅保存逐窗身份和支持，不复制巨量x。', '',
      'verify命令使用逐点标量混淆计数、另一套unique/bincount同分AP实现，并将全部严格验证和预先固定的训练首批256及原始末批≤256回读到原predict_windows；部分缺失首批回读到原tensor_input同一网络。复核拒绝不完整矩阵、重复窗口、NaN概率和改阈值。所有原始允许输入hash/mtime必须不变；seal保存两遍只读复核的交付hash/mtime不变证据。', '',
      '读取范围以config.json逐文件白名单为准。仅核对Task18 289项manifest头部及本轮白名单内文件的对应hash，不读其他epoch或其他实验的成绩。最初定位曾递归枚举其他实验文件名，未打开其数值文件；随后全部身份、推理和复核均限制到白名单。']
    return '\n'.join(lines)+'\n'


def output_snapshot(run):
    return snapshot(sorted(p for p in run.rglob('*') if p.is_file()))


def seal(run):
    require(not (run/'DELIVERY_REPORT.zh.md').exists(), 'Completed report cannot be overwritten')
    before=output_snapshot(run)
    one=verify(run)
    require(before==output_snapshot(run), 'First verify mutated delivery')
    two=verify(run)
    require(one==two and before==output_snapshot(run), 'Second verify mutated delivery')
    fresh(run/'independent_verification.json',one)
    fresh(run/'readonly_verification.json',dict(passes=2,hash_mtime_unchanged=True, files_before=len(before), source_inputs_unchanged=True))
    report = report_text(run)
    with (run/'DELIVERY_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(report)
    before=output_snapshot(run)
    require(report_text(run)==(run/'DELIVERY_REPORT.zh.md').read_text(encoding='utf-8'), 'Report readback differs')
    require(before==output_snapshot(run), 'Report function mutated output')
    fresh(run/'delivery_manifest.json',dict(status='complete_fixed_diagnostic',new_fits=0,optimizer_updates=0,
          artifacts={str(Path(p).relative_to(run)):v for p,v in before.items()},
          report_readonly_hash_mtime_unchanged=True))
    print(json.dumps(one,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('prepare','execute','verify','report','seal'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.output.resolve()
    if a.action=='prepare': prepare(run)
    elif a.action=='execute': execute(run)
    else:
        isolation(run, readonly=a.action in ('verify','report'))
        if a.action=='verify':
            before=output_snapshot(run);result=verify(run)
            require(before==output_snapshot(run),'Readonly verify mutated files')
            print(json.dumps(result,ensure_ascii=False))
        elif a.action=='report':
            before=output_snapshot(run);result=report_text(run)
            require(before==output_snapshot(run),'Readonly report mutated files')
            print(result)
        else: seal(run)


if __name__=='__main__': main()
