"""Task15 source-only feasibility audit. No training or development-data route."""
import csv
import contextlib
from pathlib import Path
import shutil

import numpy as np

from ..data.common import DataError, read_json, write_json, sha256, utc, iso
from .microwave_event_gate import hash_array
from .microwave_source5_loss import access_guard

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(read_json(PACKAGE_ROOT/'workspace_identity.json')['root']) if (PACKAGE_ROOT/'workspace_identity.json').exists() else PACKAGE_ROOT
PARENT = ROOT/'outputs/accuracy/microwave-source5-temporal-20261025-v2'
CONTROL = ROOT/'outputs/accuracy/microwave-recall-guard95-preparation-20261025-v2'
LOSS = ROOT/'outputs/accuracy/microwave-source5-loss-preparation-20261025-v4'
PLAN = ROOT/'configs/microwave_source3_background_20261025.disabled.json'
SEEDS = (42, 1019, 2026)
OFFSETS = np.arange(-63, 64, dtype=np.int64)*8


def require(value, message):
    if not value:
        raise DataError(message)


def role_values(rows, house, roles, value_field='power_w'):
    """Filter timestamps to training roles BEFORE accessing any power value."""
    result = {}
    for row in rows:
        require(row['household_id'] == house, 'Wrong household')
        t = utc(row['timestamp'])
        if not any(low <= t < high for _, low, high in roles):
            continue
        require(t % 8 == 0 and t not in result, 'Duplicate or unaligned timestamp')
        result[t] = float(row[value_field]) if row[value_field] else np.nan
    return result


@contextlib.contextmanager
def logged_access(run, allowed):
    with access_guard(run, 'prepare', allowed) as state:
        try:
            yield state
        finally:
            write_json(Path(run)/'prepare_access.json',dict(reads=sorted(state['reads']),forbidden=state['forbidden']))


def eligibility(d, micro, aggregate, roles):
    """Pure source pool builder; no interpolation, clipping or missing-value repair."""
    n = len(d['t'])
    require(d['x'].shape == (n, 127), 'Expected frozen C127')
    require(np.array_equal(np.isfinite(d['x']), d['mask']), 'Mask identity differs')
    points = d['t'][:, None] + OFFSETS
    label = np.array([[micro.get(int(t), np.nan) for t in row] for row in points], np.float32)
    raw = np.array([[aggregate.get(int(t), np.nan) for t in row] for row in points], np.float32)
    m = d['mask']
    np.testing.assert_array_equal(d['x'][m], raw[m])
    np.testing.assert_array_equal(d['y'], label[:, 63])
    known = np.isfinite(label) & (label >= 0)
    role = np.full(n, -1, np.int64)
    full_role = np.zeros(n, bool)
    observed_role = np.zeros(n, bool)
    for block, low, high in roles:
        center = (d['t'] >= low) & (d['t'] < high)
        role[center] = block
        legal = (points >= low) & (points < high)
        full_role |= center & legal.all(1)
        observed_role |= center & (legal | ~m).all(1)
    require((role >= 0).all(), 'Frozen center outside permitted training roles')
    residual = np.full(label.shape, np.nan, np.float32)
    np.subtract(d['x'], label, out=residual, where=m)
    checks = dict(template_outside_full_role=~full_role,
        template_unknown_or_negative_label=~known.all(1),
        background_observed_outside_role=~observed_role,
        background_observed_unknown_or_negative_label=~(known | ~m).all(1),
        background_invalid_residual=~((np.isfinite(residual) & (residual >= 0)) | ~m).all(1))
    pos = np.isfinite(d['y']) & (d['y'] >= 200)
    off = np.isfinite(d['y']) & (d['y'] >= 0) & (d['y'] < 200)
    template = pos & full_role & known.all(1)
    background = off & observed_role & (known | ~m).all(1) & ((np.isfinite(residual) & (residual >= 0)) | ~m).all(1)
    return dict(template=template, background=background, role=role, label=label,
                residual=residual, known_count=known.sum(1), checks=checks)


def distribution(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return dict(n=int(len(v)), quantiles=[float(x) for x in np.quantile(v, [0,.1,.5,.9,1])]) if len(v) else dict(n=0, quantiles=[])


def prepare(run):
    run = Path(run)
    require(not run.exists(), 'Preparation directory cannot be overwritten')
    p = read_json(PLAN)
    require(p['authorized'] is False and p['augmentation_rng_namespace'] == 150025, 'Disabled plan changed')
    src = read_json(PARENT/'source_identity.json')
    src['refit_5'] = read_json(PARENT/'source5_inputs.json')
    parts = read_json(ROOT/'outputs/accuracy/microwave-mask-20261025-v2/audit_manifest.json')['sources']
    roles = {part['household_id']: [(0, utc(part['start']), utc(part['end']))] for part in parts}
    roles['refit_5'] = [(i, utc(a), utc(b)) for i,a,b in read_json(PARENT/'plan.json')['house5_training_blocks']]
    allowed = [PARENT/'train.npz'] + [Path(item['path']) for h in src.values() for item in h.values()]
    for seed in SEEDS:
        allowed += [PARENT/f'sampling_{seed}.npz', PARENT/f'initial_{seed}.pt']
    # The identity's source-only allowlist contains exact model filenames.
    control_identity = read_json(CONTROL/'identity.json')
    bound_models = {path: digest for path,digest in control_identity['source_inputs'].items()
                    if str(LOSS.resolve()).lower() in path.lower() and '/models/' in path.replace('\\','/')}
    allowed += [Path(path) for path in bound_models]
    run.mkdir(parents=True)
    with logged_access(run, allowed) as access:
        for h in src.values():
            for item in h.values():
                require(sha256(Path(item['path'])) == item['sha256'], 'Source input identity changed')
        with np.load(PARENT/'train.npz', allow_pickle=False) as z:
            d = {k:z[k] for k in ('house','t','x','mask','y','event','raw_index')}
        require([len(d['y']), int((d['y']>=200).sum())] == [336704,2275], 'Frozen source support changed')
        identities = dict(train_sha256=sha256(PARENT/'train.npz'), fields={k:hash_array(v) for k,v in d.items()},
                          normalization_sha256=sha256(PARENT/'normalization.json'), source_inputs=src,
                          model_inputs={}, sampling={}, initial={}, roles=roles)
        for path,digest in bound_models.items():
            require(sha256(Path(path)) == digest, 'Frozen model input changed')
            identities['model_inputs'][path] = digest
        for seed in SEEDS:
            path = PARENT/f'sampling_{seed}.npz'
            with np.load(path, allow_pickle=False) as z:
                ix = z['indices']
            require(ix.shape == (3528,256) and ((d['y'][ix]>=200).sum(1)==16).all(), 'Frozen index quota differs')
            identities['sampling'][str(seed)] = dict(sha256=sha256(path), indices_sha256=hash_array(ix))
            identities['initial'][str(seed)] = sha256(PARENT/f'initial_{seed}.pt')
        write_json(run/'identity.json', identities)
        summaries, groups, pool_rows = [], [], []
        for hi,house in enumerate(('refit_2','refit_19','refit_5')):
            raw = {}
            for channel in ('aggregate','microwave'):
                with Path(src[house][channel]['path']).open(encoding='utf-8', newline='') as stream:
                    raw[channel] = role_values(csv.DictReader(stream), house, roles[house],
                                               'aggregate_power_w' if channel=='aggregate' else 'power_w')
            indices = np.flatnonzero(d['house']==hi)
            dd = {k:v[indices] for k,v in d.items()}
            e = eligibility(dd, raw['microwave'], raw['aggregate'], roles[house])
            pos = dd['y']>=200
            row = dict(house=house, windows=len(indices), positive_centers=int(pos.sum()),
                observable_event_ids=sorted(set(dd['event'][pos].tolist())),
                templates=int(e['template'].sum()), backgrounds=int(e['background'].sum()),
                positive_with_full127_known=int((pos & (e['known_count']==127)).sum()),
                positive_lost_due_unknown127=int((pos & (e['known_count']<127)).sum()),
                positive_known_label_points=distribution(e['known_count'][pos]),
                positive_center_w=distribution(dd['y'][pos]),
                aggregate_coverage=distribution(dd['mask'].sum(1)/127),
                valid_background_residual_w=distribution(e['residual'][e['background']]),
                rejection_counts={k:int((v & (pos if k.startswith('template') else ~pos)).sum()) for k,v in e['checks'].items()})
            summaries.append(row)
            days = np.array([iso(int(t))[:10] for t in dd['t']])
            for day in np.unique(days):
                for event in np.unique(dd['event'][days==day]):
                    take = (days==day) & (dd['event']==event)
                    groups.append(dict(house=house, utc_date=day, event_id=int(event), windows=int(take.sum()),
                        templates=int((take&e['template']).sum()), backgrounds=int((take&e['background']).sum()),
                        positive_centers=int((take&pos).sum()),
                        unknown127_positive=int((take&pos&(e['known_count']<127)).sum())))
            np.savez_compressed(run/f'audit_{house}.npz', source_index=indices, t=dd['t'], event=dd['event'],
                role=e['role'], template=e['template'], background=e['background'], known_count=e['known_count'], **e['checks'])
            for kind in ('template','background'):
                chosen = np.flatnonzero(e[kind])
                payload = e['label'][chosen] if kind=='template' else e['residual'][chosen]
                np.savez_compressed(run/f'{kind}_{house}.npz', source_index=indices[chosen], micro_or_residual=payload,
                    t=dd['t'][chosen], role=e['role'][chosen], event=dd['event'][chosen], mask=dd['mask'][chosen])
                for j,payload_row in zip(chosen,payload):
                    pool_rows.append(dict(kind=kind,house=house,utc_center=iso(int(dd['t'][j])),role=int(e['role'][j]),
                        source_index=int(indices[j]),event_id=int(dd['event'][j]),
                        window_sha256=hash_array(payload_row),mask_sha256=hash_array(dd['mask'][j])))
            print(house, row['templates'], row['backgrounds'], flush=True)
        feasible = sum(r['templates']>0 for r in summaries)>=2 and sum(r['backgrounds'] for r in summaries)>0
        audit = dict(status='feasible' if feasible else 'source_rejected', feasible=bool(feasible), houses=summaries,
            template_source_households=sum(r['templates']>0 for r in summaries),
            templates=sum(r['templates'] for r in summaries), backgrounds=sum(r['backgrounds'] for r in summaries),
            rejection='Fixed source pool fails minimum two template households or nonempty background pool' if not feasible else None,
            real_fits=0, real_updates=0, development_reads=0, final_house_reads=0,
            window_semantics='raw W; observed values verified exact against source aggregate CSV',
            event_semantics='observable threshold segments, not independent appliance cycles',
            validation_label_values_used=False)
        write_json(run/'source_pool_audit.json', audit)
        for name,rows in [('date_event_support.csv',groups),('pool_manifest.csv',pool_rows)]:
            with (run/name).open('w',encoding='utf-8',newline='') as stream:
                fields = list(rows[0]) if rows else ['kind','house','utc_center','role','source_index','event_id','window_sha256','mask_sha256']
                writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    for path in (PLAN,Path(__file__),ROOT/'scripts/run_microwave_source3_background.py',ROOT/'tests/test_microwave_source3_background.py',ROOT/'docs/MICROWAVE_SOURCE3_BACKGROUND_PLAN_20261025.md'):
        dst=run/path.relative_to(ROOT); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dst)
    write_json(run/'registration.json',dict(authorized=False,actual_real_fits=0,actual_updates=0,
        status=audit['status'],artifacts={str(path.relative_to(run)).replace('\\','/'):sha256(path) for path in run.rglob('*') if path.is_file()}))
    return report(run)


def verify(run):
    run=Path(run)
    reg=read_json(run/'registration.json')
    require(reg['authorized'] is False and reg['actual_real_fits']==0 and reg['actual_updates']==0, 'Disabled state changed')
    for name,digest in reg['artifacts'].items():
        require(sha256(run/name)==digest,'Preparation artifact changed: '+name)
    return read_json(run/'source_pool_audit.json')


def report(run):
    a=verify(run)
    dest=Path(run)/'PREPARATION_REPORT.md'
    if not dest.exists():
        lines=['# 三来源背景组合可行性审计', '',f"结论：{a['status']}。固定资格下模板 {a['templates']} 个，背景 {a['backgrounds']} 个；模板来源家庭 {a['template_source_households']} 个。",'',
            '真实训练 0 次，参数更新 0 次。未读取 House17 数值、预测、指标、真值或最终 8/9/20 数据。', '',
            '|来源|原正中心|127点全知正中心|合法模板|合法背景|因127点未知丢失正中心|', '|---|---:|---:|---:|---:|---:|']
        for r in a['houses']:
            lines.append(f"|{r['house']}|{r['positive_centers']}|{r['positive_with_full127_known']}|{r['templates']}|{r['backgrounds']}|{r['positive_lost_due_unknown127']}|")
        lines += ['', '未知标签保留未知；未补零、插值、缩窗或放宽角色限制。缓存原始W与来源CSV在全部可观察点逐值相等。', '',
            '日期、可观察事件ID逐组支持见 date_event_support.csv；逐中心拒绝布尔矩阵见 audit_refit_*.npz；功率/覆盖分布和拒绝计数见 source_pool_audit.json。窗口与阈值片段不等于独立运行周期。', '',
            '固定策略预登记为每批前8个正slot替换、独立RNG 150025、家庭/UTC日→事件→中心分层模板抽样及均匀背景抽样。旧索引曝光与模板实际曝光必须分开记录。', '',
            ('可行性通过。完整实现准备见更新的不可覆盖实施快照；本目录只封存来源池。真实训练仍需本任务额外明确人类授权。' if a['feasible'] else '可行性失败，来源拒绝；不训练、不放宽，需另选方案。'), '',
            '本阶段不主张增强有效，也不主张独立泛化；House17 若未来使用仅作工程开发。UI默认与功率输出未改。']
        dest.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return a


def pending(run, stage, authorization=None):
    a=verify(run)
    raise DataError(f"{stage} disabled: authorized=false; source status={a['status']}; no Task15 training adapter or authorization receipt exists")


def complete_prepare(run,pool_run):
    """Reuse sealed v2 artifacts byte for byte; do not rebuild or reopen raw curves."""
    from . import microwave_source3_background_execution as execute
    from . import microwave_recall_guard95 as guard95
    from . import microwave_event_gate as gate
    from .microwave_sampling import BatchStream
    import torch
    run,pool_run=Path(run),Path(pool_run).resolve()
    require(not run.exists(),'New implementation preparation directory required')
    pool=read_json(pool_run/'source_pool_audit.json')
    require(pool['feasible'],'Cannot prepare training for infeasible pools')
    run.mkdir(parents=True)
    ident=read_json(pool_run/'identity.json')
    allowed=[PARENT/'train.npz']+[PARENT/f'sampling_{s}.npz' for s in SEEDS]+[PARENT/f'initial_{s}.pt' for s in SEEDS]
    allowed += [LOSS/f'models/unit_BCE/seed_{s}/initial.pt' for s in SEEDS]
    allowed += [pool_run/k for k in read_json(pool_run/'registration.json')['artifacts'] if k.endswith(('.npz','.csv'))]
    allowed += [Path(p) for p in ident['model_inputs']]
    with logged_access(run,allowed):
        verify(pool_run)
        # Reuse existing numerical pools and audit evidence without recomputing them.
        reused={}
        for path in pool_run.iterdir():
            if path.name.startswith(('audit_refit_','template_refit_','background_refit_')) or path.name in ('date_event_support.csv','pool_manifest.csv','source_pool_audit.json'):
                shutil.copyfile(path,run/path.name)
                require(sha256(run/path.name)==sha256(path),'Reused source artifact differs')
                reused[path.name]=sha256(path)
        ident['reused_pool']=dict(path=str(pool_run),registration_sha256=sha256(pool_run/'registration.json'),artifacts=reused)
        ident['source_metadata']={str(CONTROL/name):sha256(CONTROL/name) for name in ('source_selection.json','source_complete.json','registration.json')}
        selected=read_json(CONTROL/'source_selection.json')
        require(selected['source_only'] and selected['selected']=='guard95' and selected['guard95_eligible'],'Expected frozen Task14 source-selected control')
        complete=read_json(CONTROL/'source_complete.json')
        require(complete['source_selection_sha256']==sha256(CONTROL/'source_selection.json') and complete['registration_sha256']==sha256(CONTROL/'registration.json'),'Control source seal differs')
        require(selected['algorithm_sha256']==sha256(Path(guard95.__file__)),'Task14 fixed algorithm differs')
        ident['algorithms']=dict(gate_sha256=sha256(Path(gate.__file__)),guard95_sha256=sha256(Path(guard95.__file__)))
        ident['source_curve_identities_reused_without_read']=True
        norm=read_json(PARENT/'normalization.json')
        p=read_json(PARENT/'plan.json')
        # Only small center identity arrays are decompressed; no repeated x/CSV read.
        with np.load(PARENT/'train.npz',allow_pickle=False) as z:
            centers={k:z[k] for k in ('house','t','event','y')}
        proof={}
        for seed in SEEDS:
            stream=BatchStream(centers,'event',seed,p)
            with np.load(PARENT/f'sampling_{seed}.npz',allow_pickle=False) as z: ix=z['indices']
            for row in ix: np.testing.assert_array_equal(row,stream.next())
            initial=torch.load(PARENT/f'initial_{seed}.pt',weights_only=True,map_location='cpu')
            parent_initial=LOSS/f'models/unit_BCE/seed_{seed}/initial.pt'
            previous=torch.load(parent_initial,weights_only=True,map_location='cpu')
            require(initial.keys()==previous.keys() and all(torch.equal(initial[k],previous[k]) for k in initial),'Control/aug initial weights differ')
            meta=read_json(LOSS/f'models/unit_BCE/seed_{seed}/model.json')
            require(all(meta[k]==v for k,v in norm.items()) and meta['recipe']=='unit_BCE','Frozen model normalization/loss differs')
            proof[str(seed)]=dict(frozen_indices_exact_replay=True,batches=3528,original_positive_slots=16,
                initial_control_exact=True,initial_tensor_hashes={k:hash_array(v.numpy()) for k,v in initial.items()},
                initial_source_sha256=sha256(parent_initial))
            ident['model_inputs'][str(parent_initial)]=sha256(parent_initial)
        original_seconds={str(seed):read_json(LOSS/f'models/unit_BCE/seed_{seed}/complete.json')['seconds'] for seed in SEEDS}
        write_json(run/'cost_reference.json',dict(reference='frozen Task12 unit_BCE same-shape training; no new timing fit',seconds=original_seconds,
            total_seconds=sum(original_seconds.values()),maximum_additional_fits=3,maximum_additional_updates=10584,
            overhead='augmentation and logging add work; reference is not an upper bound',billable_price_not_available=True))
        write_json(run/'identity.json',ident)
        write_json(run/'frozen_replay.json',dict(seeds=proof,source_x_reopened=False,raw_csv_reopened=False,real_fits=0,real_updates=0))
        shutil.copyfile(PLAN,run/'config.json')
        shutil.copyfile(PARENT/'plan.json',run/'parent_training_plan.json')
        write_json(run/'workspace_identity.json',dict(root=str(ROOT.resolve())))
        write_json(run/'authorization_template.disabled.json',dict(authorized=False,approval_origin='human',
            experiment='microwave_source3_background_v1',registration_sha256=None,seeds=list(SEEDS),
            additional_new_fits=3,maximum_updates=10584,approved_augmentation='fixed_source_template_background_8_of_16',
            human_authorization_quote=None,human_authorization_reference=None))
    shutil.copytree(ROOT/'src/virtual_submeter',run/'source/virtual_submeter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (run/'configs').mkdir()
    shutil.copyfile(ROOT/'configs/microwave_source5_temporal_20261025.json',run/'configs/microwave_source5_temporal_20261025.json')
    shutil.copyfile(ROOT/'scripts/run_microwave_source3_background.py',run/'run_microwave_source3_background.py')
    for path in (ROOT/'tests/test_microwave_source3_background.py',ROOT/'tests/test_microwave_source3_background_execution.py',ROOT/'docs/MICROWAVE_SOURCE3_BACKGROUND_PLAN_20261025.md'):
        shutil.copyfile(path,run/path.name)
    write_json(run/'registration.json',dict(authorized=False,actual_real_fits=0,actual_updates=0,status='complete_preparation_pending_human_augmentation_budget',
        artifacts={str(path.relative_to(run)).replace('\\','/'):sha256(path) for path in run.rglob('*') if path.is_file()}))
    return execute.preparation_report(run)
