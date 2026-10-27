"""Supplement sealed Task22 v4 diagnostics with static arrays and fake inference only.

No real models, training data, optimizer, fit, backward, development or final input.
The sealed diagnostics function, input conversion, metric and output routes run as-is.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SEALED=ROOT/'outputs/accuracy/microwave-task22-preparation-20261026-v4'
sys.path.insert(0,str(SEALED/'source'))
from virtual_submeter.accuracy import microwave_task22 as m


def ensure(condition,message):
    if not condition:raise AssertionError(message)


def scalar_metric(probability,y,threshold):
    # Deliberately use no production confusion or AP helper.
    truth=[bool(v>=200) for v in y]
    counts=dict(tp=0,fp=0,fn=0,tn=0)
    for score,actual in zip(probability,truth):
        predicted=bool(score>=threshold)
        counts['tp' if predicted and actual else 'fp' if predicted else 'fn' if actual else 'tn']+=1
    tp,fp,fn,tn=[counts[k] for k in ('tp','fp','fn','tn')]
    # Sum positive mass at each distinct descending score, including entire tie group.
    groups={}
    for score,actual in zip(probability,truth):
        on,total=groups.get(float(score),(0,0));groups[float(score)]=(on+int(actual),total+1)
    found=seen=0;ap=0.
    for score in sorted(groups,reverse=True):
        on,total=groups[score];found+=on;seen+=total
        ap+=(on/max(sum(truth),1))*(found/seen)
    return dict(**counts,n=len(y),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
        fpr=fp/max(fp+tn,1),predicted_on=tp+fp,degenerate=(tp+fp) in (0,len(y)),ap=ap,threshold=float(threshold))


class FakeNet(torch.nn.Module):
    """No parameters or learned state; width is an explicit simulated model identity."""
    def __init__(self,channels,seed,calls):
        super().__init__();self.channels=channels;self.seed=seed;self.calls=calls
    def forward(self,z):
        ensure(z.ndim==3 and z.shape[1:]==(2,127),'C127/mask input missing')
        ensure(torch.get_num_threads()==4,'Metadata CPU4 environment not applied')
        ensure(not torch.is_grad_enabled(),'Autograd unexpectedly enabled')
        self.calls.append(dict(channels=self.channels,seed=self.seed,batch_size=len(z),cpu_threads=torch.get_num_threads(),
            partial_core_rows=int((z[:,1,48:79].sum(1)<31).sum())))
        # Tied logits depend on center value and one actual observation-mask bit.
        return z[:,0,63]*.5+z[:,1,48]*.25+(self.channels-16)/64+(self.seed%7)*.01-1.2


def main():
    ensure(not (HERE/'supplement_manifest.json').exists(),'Completed supplement is immutable')
    ensure(m.__file__==str(SEALED/'source/virtual_submeter/accuracy/microwave_task22.py'),'Not using sealed v4 diagnostics')
    old_v2=ROOT/'outputs/accuracy/microwave-task22-preparation-20261026-v2'
    preserved=[p for p in SEALED.rglob('*') if p.is_file()]+[p for p in old_v2.rglob('*') if p.is_file()]+[old_v2.with_name(old_v2.name+'.readonly-receipt.json')]
    target_receipt=SEALED.with_name(SEALED.name+'.readonly-receipt.json')
    if target_receipt.exists():preserved.append(target_receipt)
    sealed_before=m.snapshot(preserved)
    registration=m.sha256(SEALED/'registration.json')
    ensure(registration=='2ffd72773185d8be1372b0744729b7eb3daf8656f1dfd2097da47cb113037c53','v4 registration differs')
    run=HERE/'fixture_run';t10=HERE/'static_t10';t12=HERE/'static_t12'
    run.mkdir(exist_ok=False);(t10/'validation').mkdir(parents=True,exist_ok=False)
    # Both labels occur within both train views. No missing labels are filled.
    def dataset(n,offset,house,partial):
        x=np.full((n,127),100.,np.float32)
        powers=np.array([0,10,100,300,700,1500],np.float32)
        x[:,63]=powers[np.arange(n)%6]
        y=np.where(np.arange(n)%3==0,300.,0.).astype(np.float32)
        x[1::5,:20]=np.nan;x[3::7,100:]=np.nan
        if partial:
            x[::4,48]=np.nan;x[2::9,62]=np.nan
        return dict(x=x,mask=np.isfinite(x),y=y,t=offset+np.arange(n,dtype=np.int64)*8,
            house=np.full(n,house,np.int64),event=np.where(y>=200,np.arange(n)//3,-1))
    datasets={'train':dataset(41,1300000000,0,True),
        'block_4':dataset(23,1416743880,2,False),'block_5':dataset(19,1416916680,2,False)}
    np.savez_compressed(t10/'train.npz',**datasets['train'])
    for b in (4,5):np.savez_compressed(t10/f'validation/block_{b}.npz',**datasets[f'block_{b}'])
    datasets['source']={k:np.concatenate([datasets[f'block_{b}'][k] for b in (4,5)]) for k in ('x','y','t','house')}
    ensure(not np.isfinite(datasets['train']['x'][:,48:79]).all(1).all(),'Train fixture needs partial cores')
    ensure(np.isfinite(datasets['source']['x'][:,48:79]).all(),'Source fixture must retain strict cores')
    calls=[];loads=[];auth_calls=[];numerical_reads=[]
    meta=dict(input_log_mean=5.,input_log_std=.8,cpu_threads=4,batch_size=7,window_samples=127,
        inference_support='strict_complete_31',truth_threshold_w=200,decision_threshold_probability=.5)
    # Explicit fake identity files are hashed by the real completion/registration guards.
    for seed in m.SEEDS:
        for arm,channels,folder in (('control16',16,t12/f'models/unit_BCE/seed_{seed}'),('wide32',32,run/f'models/wide32/seed_{seed}')):
            folder.mkdir(parents=True,exist_ok=False)
            m.fresh(folder/'model.json',dict(**meta,tcn_channels=channels,parameter_count=5329 if channels==16 else 19809,
                seed=seed,synthetic_fake_model=True,trained=False))
            with (folder/'weights.pt').open('xb') as f:f.write(b'static-fixture-placeholder-not-torch-weights')
    m.fresh(run/'training_complete.json',dict(new_fits=3,updates=10584,control_retrains=0,
        synthetic_simulated_completion=True,actual_real_fits=0,actual_updates=0,artifacts=m.inventory(run/'models')))
    m.fresh(run/'config.json',m.read_json(SEALED/'config.json'))
    inputs=[*t10.rglob('*'),*t12.rglob('*'),*list((SEALED/'source').rglob('*.py')),Path(__file__).resolve()]
    inputs=[p.resolve() for p in inputs if p.is_file()]
    m.fresh(run/'inputs_before.json',m.snapshot(inputs))
    m.fresh(run/'registration.json',dict(schema=m.SCHEMA,authorized=False,synthetic_fixture_only=True,artifacts=m.inventory(run)))
    def fake_authorize(path,receipt):
        ensure(Path(path)==run and receipt=='static-fixture-no-human-budget','Unexpected authorization target')
        auth_calls.append(dict(path=str(path),synthetic_only=True));return dict(synthetic_only=True,real_budget=False)
    def fake_load(path):
        path=Path(path);info=m.read_json(path/'model.json')
        ensure(info['synthetic_fake_model'] is True and info['trained'] is False,'Attempt to load a real model')
        loads.append(dict(path=str(path),channels=info['tcn_channels'],parameter_count=info['parameter_count'],seed=info['seed']))
        return info,FakeNet(info['tcn_channels'],info['seed'],calls).eval()
    real_load=m.load
    def record_load(path,fields=None):
        numerical_reads.append(str(Path(path).resolve()));return real_load(path,fields)
    def no_training(*args,**kwargs):raise AssertionError('Optimizer, training and autograd forbidden')
    torch.set_num_threads(4)
    run_before=m.snapshot([*inputs,*[run/k for k in m.inventory(run)]])
    # Only external auth, model-loading and fixture roots are injected. Diagnostics,
    # registered/hash guards, tensor conversion, AP/confusion, storage and reject route are real.
    with patch.object(m,'T10',t10),patch.object(m,'T12',t12),patch.object(m,'authorize',side_effect=fake_authorize),\
            patch.object(m,'load_state_model',side_effect=fake_load),patch.object(m,'load',side_effect=record_load),\
            patch.object(torch.optim.Optimizer,'__init__',no_training),patch.object(torch.optim.Optimizer,'add_param_group',no_training),\
            patch.object(torch.Tensor,'backward',no_training),patch.object(torch.autograd,'backward',no_training),\
            patch.object(torch.autograd,'grad',no_training),patch.object(m.loss.temporal,'fit_one',no_training):
        with m.isolation(run,m.execution_allowed(run)) as access:
            result=m.diagnostics(run,'static-fixture-no-human-budget')
        read_access=dict(reads=sorted(access['reads']),forbidden=access['forbidden'])
        ensure(result==dict(diagnostic_cells=24,development_reads=0,new_selections=0),'Diagnostic route result differs')
        output_before=m.snapshot([run/k for k in m.inventory(run)])
        loads_before=len(loads);calls_before=len(calls);reads_before=len(numerical_reads)
        try:m.diagnostics(run,'static-fixture-no-human-budget')
        except m.DataError as exc:
            ensure('cannot overwrite' in str(exc),'Wrong duplicate rejection');duplicate_reason=str(exc)
        else:raise AssertionError('Duplicate diagnostics accepted')
        ensure(len(loads)==loads_before and len(calls)==calls_before and len(numerical_reads)==reads_before,'Duplicate opened model/data or inferred')
        ensure(m.snapshot(output_before)==output_before,'Duplicate changed outputs')
    m.gate.check(run/'diagnostics',m.read_json(run/'diagnostics_complete.json')['artifacts'])
    metrics=m.read_json(run/'diagnostics/metrics.json')
    ensure(metrics['explanatory_only'] and metrics['new_thresholds']==metrics['new_selections']==metrics['development_reads']==0,'Selection or threshold change')
    rows=metrics['rows'];ensure(len(rows)==24,'Expected 24 native cells')
    expected={(arm,seed,data,view) for arm in ('control16','wide32') for seed in m.SEEDS for data in ('train','source') for view in ('original','strict_core31')}
    ensure({(r['arm'],r['seed'],r['dataset'],r['view']) for r in rows}==expected,'Cell matrix missing/duplicated')
    caches=sorted((run/'diagnostics').glob('*.npz'));ensure(len(caches)==12,'Expected 12 probability caches')
    comparisons=[];support_identity={};ties_verified=False
    for arm in ('control16','wide32'):
        for seed in m.SEEDS:
            for name in ('train','source'):
                d=datasets[name];cache=real_load(run/f'diagnostics/{arm}_{seed}_{name}.npz')
                np.testing.assert_array_equal(cache['t'],d['t']);np.testing.assert_array_equal(cache['house'],d['house'])
                ensure(cache['probability'].shape==(len(d['y']),) and np.isfinite(cache['probability']).all(),'Cached probability support differs')
                # Independent scalar input conversion and fake-forward formula, preserving
                # NumPy float32/Torch operation order to require exact float32 probabilities.
                center=d['x'][:,63];z=np.zeros(len(center),np.float32);observed=np.isfinite(center)
                z[observed]=(np.log1p(center[observed])-meta['input_log_mean'])/meta['input_log_std']
                logits=torch.from_numpy(z)*.5+torch.from_numpy(np.isfinite(d['x'][:,48]).astype(np.float32))*.25+((16 if arm=='wide32' else 0)/64)+(seed%7)*.01-1.2
                expected_probability=np.concatenate([torch.sigmoid(v).numpy() for v in logits.split(meta['batch_size'])])
                np.testing.assert_array_equal(cache['probability'],expected_probability)
                ties_verified |= len(np.unique(cache['probability']))<len(cache['probability'])
                for view in ('original','strict_core31'):
                    mask=np.ones(len(d['y']),bool) if view=='original' else np.isfinite(d['x'][:,48:79]).all(1)
                    row=next(r for r in rows if (r['arm'],r['seed'],r['dataset'],r['view'])==(arm,seed,name,view))
                    ensure(row['support_sha256']==m.gate.hash_array(mask) and row['n']==int(mask.sum()),'View support differs')
                    key=(name,view);identity=(m.gate.hash_array(d['t'][mask]),row['support_sha256'],row['n'])
                    ensure(support_identity.setdefault(key,identity)==identity,'Control/new rows not same-window aligned')
                    independent=scalar_metric(cache['probability'][mask],d['y'][mask],meta['decision_threshold_probability'])
                    for k,value in independent.items():
                        ensure(abs(row[k]-value)<2e-12 if isinstance(value,float) else row[k]==value,'Independent scalar/AP differs: '+k)
                    comparisons.append(dict(arm=arm,seed=seed,dataset=name,view=view,passed=True,independent=independent))
    ensure(ties_verified,'Fixture lacks tied probabilities')
    ensure(support_identity[('train','strict_core31')][2]<support_identity[('train','original')][2],'Partial cores did not reduce strict train support')
    ensure(support_identity[('source','strict_core31')]==support_identity[('source','original')],'Strict source support changed')
    ensure({v['channels'] for v in loads}=={16,32} and len(loads)==6,'Fake16/32 model matrix differs')
    ensure(any(v['partial_core_rows']>0 for v in calls),'Training partial core never forwarded')
    ensure(m.snapshot(run_before)==run_before,'Supplement changed pre-existing fixture inputs')
    ensure(m.snapshot(sealed_before)==sealed_before,'Sealed v2 hash/mtime changed')
    observed_paths=numerical_reads+[v['path'] for v in loads]
    ensure(all(str(HERE) in p for p in observed_paths),'External real data/model path read')
    ensure(not read_access['forbidden'],'Whitelist violation')
    m.fresh(HERE/'independent_comparisons.json',dict(cells=comparisons,cell_count=24,alignment_across_arms_and_seeds=True,
        support={f'{a}/{b}':dict(timestamp_sha256=v[0],mask_sha256=v[1],n=v[2]) for (a,b),v in support_identity.items()}))
    m.fresh(HERE/'access_receipt.json',dict(**read_access,numerical_reads=numerical_reads,fake_model_loads=loads,forward_calls=calls,
        new_fits=0,optimizers=0,optimizer_updates=0,backward_calls=0,development_reads=0,final_reads=0,cpu_threads=4,
        patched_boundaries=['synthetic authorization only','fixture T10/T12 roots only','fake model loader only','read auditing wrapper','optimizer/autograd/fit blockers'],
        diagnostics_function_unmodified=True,registration_guard_unmodified=True))
    m.fresh(HERE/'verification_receipt.json',dict(passed=True,old_v2_registration_sha256=m.sha256(old_v2/'registration.json'),old_v2_unchanged=True,v4_registration_sha256=registration,native_cells=24,prediction_caches=12,
        duplicate_rejected=True,duplicate_reason=duplicate_reason,duplicate_model_data_reads=0,duplicate_forwards=0,
        fixture_inputs_hash_mtime_unchanged=True,sealed_v4_hash_mtime_unchanged=True,cpu_threads_explicitly_set=4,
        same_window_AP_and_scalar_counts_exact=True,AP_absolute_tolerance=2e-12,tied_probabilities_tested=True,
        train_partial_core_and_source_strict_tested=True,new_fits=0,optimizers=0,optimizer_updates=0,backward_calls=0,
        development_reads=0,final_reads=0,sealed_v4_before=sealed_before,fixture_before=run_before))
    report='''# Task22 v4 diagnostics 补充验证

结论：封存 v4 的 diagnostics 函数已用独立静态数组和假模型实际运行通过。既有口径为 **24 格指标、12 份预测缓存**，未增加分组或改变源码。

矩阵为2模型族(control16/wide32)×3seed(42/1019/2026)×2数据集(train/source合并)×2支持视图(original/strict_core31)。训练夹具有部分core31缺失，来源block4/5均保留严格core31；严格训练支持小于原支持，来源两视图支持完全一致。各模型族与种子的同窗时间/家庭/支持对齐逐项核对。

显式设置CPU线程为4，并在每次假模型forward中断言。封存代码未为此修改。假模型没有可训练参数、没有真实权重，输入转换仍执行封存tensor_input；缓存概率按独立输入公式逐点exact，所有24格AP和固定阈值混淆计数/FPR/P/R/F1以独立标量实现复算，AP同分组处理误差小于2e-12。

真实运行了diagnostics的注册/完成清单检查、完整循环、指标、12份缓存、完成封存和重复拒绝分支；仅注入静态路径、明确假授权与假模型加载。重复调用在模型/数据加载、forward和任何输出写入前拒绝；已有输出hash/mtime不变。

0真实fit、0优化器、0更新、0反传、0开发/最终读取。所有数值和模型读取均在本补充夹具目录内；封存v4全目录及原只读回执hash/mtime前后完全一致。新训练预算仍未授权。

v4注册SHA：`'''+registration+'''`
'''
    report+='\n冷启动发现v2的diagnostics在白名单内延迟导入复核模块，Python探测未登记pyc导致拒绝。v2保留原SHA且未改动，失败证据在补充v1；v3首次导入修复另遇复核模块优先加载的循环依赖并保留未封存失败记录。v4将独立AP移入无协议依赖的数值模块，进入限制前加载数值帮助函数；不改变24格诊断口径。本次从v4快照冷启动运行通过。\n'
    with (HERE/'SUPPLEMENT_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write(report)
    m.fresh(HERE/'supplement_manifest.json',dict(status='passed_static_diagnostics_no_real_fit',v4_registration_sha256=registration,
        artifacts=m.inventory(HERE),new_fits=0,optimizer_updates=0))
    print(json.dumps(dict(passed=True,native_cells=24,prediction_caches=12,cpu_threads=4,v4_registration_sha256=registration,
        v4_unchanged=True,new_fits=0,optimizer_updates=0,development_reads=0,final_reads=0),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
