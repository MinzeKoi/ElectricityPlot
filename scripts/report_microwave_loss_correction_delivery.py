"""Finalize the Chinese task 8 report and complete artifact hash receipt."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run.resolve();root=Path(__file__).resolve().parents[1]
    review=read(run/'independent_delivery_verification.json')
    assert review['status']=='independently_verified_completed_delivery'
    assert digest(run/'freeze.json')==review['freeze_sha256']
    summary=read(run/'evaluation/summary.json')
    text=(run/'report.md').read_text(encoding='utf-8').replace('0.14084224','0.14083946')
    text+='\n## 逐seed误报配对差异\n\n| 组 | seed | 对照 | ΔP | ΔR | ΔF1 | ΔFP | FP减少比例 |\n|---|---:|---|---:|---:|---:|---:|---:|\n'
    with (run/'paired_false_positive_comparison.csv').open(encoding='utf-8',newline='') as stream:
        for r in csv.DictReader(stream):
            text+=f"| {r['arm']} | {r['seed']} | {r['reference']} | {float(r['precision_delta']):+.8f} | {float(r['recall_delta']):+.8f} | {float(r['f1_delta']):+.8f} | {int(r['fp_delta']):+d} | {float(r['fp_reduction']):.2%} |\n"
    text+='\n## 实际损失与有限循环\n\n| 组 | seed | 正损失总量 | 负损失总量 | 正损失占比 | 实际名义正权质量 |\n|---|---:|---:|---:|---:|---:|\n'
    training=[]
    for arm in ['importance_BCE50','unit_BCE']:
        for seed in (42,1019,2026):
            path=run/f'models/{arm}/seed_{seed}'
            with (path/'batches.csv').open(encoding='utf-8',newline='') as stream: batches=list(csv.DictReader(stream))
            pl=sum(float(b['positive_loss_sum']) for b in batches);nl=sum(float(b['negative_loss_sum']) for b in batches)
            nominal=np.mean([float(b['nominal_positive_weight_mass']) for b in batches])
            text+=f'| {arm} | {seed} | {pl:.6f} | {nl:.6f} | {pl/(pl+nl):.8f} | {nominal:.8f} |\n'
            training.append(dict(arm=arm,seed=seed,positive_loss_sum=pl,negative_loss_sum=nl,positive_loss_fraction=pl/(pl+nl),mean_batch_nominal_positive_mass=float(nominal)))
    # Empirical finite-prefix center/class exposure under the registered hierarchy.
    with np.load(run/'weights.npz',allow_pickle=False) as z: prob=z['probability'];positive=z['positive'];importance=z['importance']
    finite=[]
    for seed in (42,1019,2026):
        with np.load(run/f'sampling_{seed}.npz',allow_pickle=False) as z: idx=z['indices']
        exposure=np.bincount(idx.ravel(),minlength=len(prob));empirical=exposure/idx.size
        finite.append(dict(seed=seed,draws=int(idx.size),positive_draws=int(exposure[positive].sum()),negative_draws=int(exposure[~positive].sum()),
            finite_prefix_total_variation_from_longrun=float(np.abs(empirical-prob).sum()/2),
            importance_min=float(importance.min()),importance_max=float(importance.max()),
            empirical_weighted_positive_mass=float((exposure[positive]*importance[positive]*50).sum()/((exposure[positive]*importance[positive]*50).sum()+(exposure[~positive]*importance[~positive]).sum()))))
    write(run/'loss_and_finite_cycle_audit.json',dict(training=training,finite_cycle=finite))
    text+='\n逐层shuffle无放回导致有限前缀中心曝光与长期边际概率不完全相等，详细总变差和实际加权正质量见loss_and_finite_cycle_audit.json。这里只主张固定logits对长期边际分布的加权期望恒等；实际训练中logits随历史变化，不主张每步条件独立无偏。\n'
    text+='\n## 复核与交付\n\n独立算术复核精确重现全部15行混淆矩阵和P/R/F1；每seed 137523评价点；9个冻结模型状态与普通加载一致；原A所有电器功率逐点不变，其他电器状态由原阈值派生也不变。6项封存损失测试、3项任务专用拒绝路径测试、21项父TCN回归测试通过（共30项）。\n'
    text+='\nv1仅有一次准备读取白名单遗漏，未启动真实拟合或更新；原目录已保留，v2修复后封存执行。测试启动器首次未指向外部配置导致6项测试定位错误，修正测试路径后全部通过，未改变模型代码或目标。新训练无失败、无额外启动。父任务294步接口失败与6有效拟合独立列账。\n'
    text+='\n完整精确源PR曲线为pr/source/下54个NPZ，开发PR为pr/development/下15个NPZ（A/W50阈值以瓦为单位，分类器阈值为概率）；development_pr.svg为开发预览。每个NPZ含thresholds/precision/recall/f1，源曲线固定各seed已选checkpoint，未据开发重新选择。\n'
    text+='\n复现入口（在项目根运行）：\n\n```powershell\n.venv-m2/Scripts/python.exe -B outputs/accuracy/microwave-loss-correction-20261025-v2/run_microwave_loss_correction.py verify --run outputs/accuracy/microwave-loss-correction-20261025-v2\n```\n\n候选可由标准center_mask_tcn.load_state_model加载，模型目录outputs/accuracy/microwave-loss-correction-20261025-v2/models/{arm}/seed_{seed}；原predict_microwave_tcn.py --model指向这些目录即可按普通接口推理。训练与评价禁止重复执行覆盖已有目录。\n'
    (run/'report.md').write_text(text,encoding='utf-8')
    shutil.copyfile(run/'report.md',root/'docs/MICROWAVE_LOSS_CORRECTION_RESULTS_20261025.md')
    write(root/'docs/MICROWAVE_LOSS_CORRECTION_RESULTS_20261025.json',dict(**summary,loss_audit=training,finite_cycle=finite,independent_review=review))
    write(run/'runtime_delivery.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__))
    files=['src/virtual_submeter/accuracy/microwave_loss_correction.py','scripts/run_microwave_loss_correction.py',
        'scripts/verify_microwave_loss_correction_delivery.py','scripts/report_microwave_loss_correction_delivery.py',
        'tests/test_microwave_loss_correction.py','tests/test_microwave_loss_correction_guards.py',
        'configs/microwave_loss_correction_20261025.json','docs/MICROWAVE_LOSS_CORRECTION_PLAN_20261025.md',
        'docs/MICROWAVE_LOSS_CORRECTION_RESULTS_20261025.md','docs/MICROWAVE_LOSS_CORRECTION_RESULTS_20261025.json',
        'tmp/microwave-loss-training-20261025.log','tmp/microwave-loss-sealed-tests-20261025.log',
        'tmp/microwave-loss-guard-tests-20261025.log','tmp/microwave-loss-parent-regression-20261025.log']
    output_hashes={str(p.relative_to(run)):digest(p) for p in run.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    receipt=dict(status='completed_accepted' if summary['development_acceptance_passed'] else 'completed_not_accepted',
        actual_run=str(run),new_fit_starts=6,new_successful_fits=6,new_updates=21168,new_failed_fits=0,reused_control_fits=3,
        parent_budget=review['parent_budget'],prepare_failures_zero_fits=1,tests_passed=30,independent_correctness_review_passed=True,
        selected_before_development=summary['selected_before_development'],development_acceptance_passed=summary['development_acceptance_passed'],
        default_model_changed=False,final_houses_accessed=False,truth_csv_parses=1,
        registration_sha256=digest(run/'registration.json'),freeze_sha256=digest(run/'freeze.json'),
        artifacts={name:digest(root/name) for name in files},run_artifacts=output_hashes)
    write(root/'docs/MICROWAVE_LOSS_CORRECTION_DELIVERY_20261025.json',receipt)
    assert all(digest(run/name)==h for name,h in output_hashes.items())
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['artifacts','run_artifacts']},ensure_ascii=False,indent=2))
