"""Seal Task24 execution evidence without training or opening development data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1'
JOB = ROOT / 'outputs/accuracy/microwave-task24-recovery-execution-20261028-v1'
PROOF = ROOT / 'outputs/accuracy/microwave-task24-execution-readonly-proof-20261028-v1'
REVIEW = ROOT / 'outputs/accuracy/microwave-task24-independent-review-20261027-v1'
INTERRUPTION = ROOT / 'outputs/accuracy/microwave-task24-interruption-audit-20261027-v1/receipt.json'
AUTH = ROOT / 'outputs/accuracy/microwave-task24-recovery-delivery-20261027-v1/human-recovery-budget-20261028.json'
DEST = ROOT / 'outputs/accuracy/microwave-task24-execution-delivery-20261028-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not DEST.exists()
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    proof = read(PROOF / 'receipt.json')
    assert proof['verify_passes'] == proof['report_passes'] == 2
    for item in proof['files']:
        assert sha(PROOF / item['name']) == item['sha256']
    summary = read(PROOF / 'verify-1.json')
    assert summary['status'] == 'source_rejected_control_retained' and summary['recovery_updates'] == 10584
    train_review = read(REVIEW / 'training_review_20261028.receipt.json')
    source_review = read(REVIEW / 'source_review_20261028.receipt.json')
    assert train_review['passed'] and source_review['passed']
    previous = read(INTERRUPTION)
    assert previous['actual_fits'] == previous['optimizer_updates'] == 0
    selection = read(RUN / 'source_selection.json')
    control = read(ROOT / 'outputs/accuracy/microwave-recall-guard95-preparation-20261025-v2/source_selection.json')['rows']['guard95']
    candidate = selection['rows']
    mean = summary['source_relative_comparison']
    logs = [JOB / n for n in ('train_process.json', 'train.stdout.log', 'train.stderr.log', 'source_process.json', 'source.stdout.log', 'source.stderr.log')]
    assert all(p.is_file() for p in logs)
    assert (JOB / 'train.stderr.log').stat().st_size == (JOB / 'source.stderr.log').stat().st_size == 0
    assert not any((RUN / n).exists() for n in ('future_identity.json', 'predictions', 'evaluation'))
    rows = '\n'.join(f"| {a['seed']} | {a['precision']:.6f} / {a['recall']:.6f} / {a['f1']:.6f} | {b['precision']:.6f} / {b['recall']:.6f} / {b['f1']:.6f} |" for a, b in zip(control, candidate))
    report = f'''# Task24 多家庭 C127 固定更新量对照：执行报告

**来源对照未通过，保留既有 control_guard95；未进入开发验收。** 三个多家庭候选在固定来源验证中的平均精度和平均 F1 均低于旧控制。三种子六个验证块的召回均达到 95%，但只有一个种子的精度提高，未满足至少两个种子提高的相对资格要求；候选平均 F1 也低于控制。按固定排序，候选平均精度同样未胜出。

| 种子 | 旧控制 P / R / F1 | 多家庭候选 P / R / F1 |
|---|---|---|
{rows}
| 三种子平均 | {mean['control_mean_precision']:.6f} / — / {mean['control_mean_f1']:.6f} | {mean['candidate_mean_precision']:.6f} / — / {mean['candidate_mean_f1']:.6f} |

表中各种子指标是两个固定 House5 来源验证块的平均值。种子 42 的候选 F1 上升；下降的是三种子平均 F1。各块召回和逐项混淆计数见独立复核报告。本次结果仅适用于原 16 通道 C127、unit_BCE、固定采样与固定更新量的对照。数据扩池同时改变了训练分布和每个中心的曝光，不能将差异唯一归因于家庭多样性；旧多户功率回归扩数也已有负结果。

## 预算和执行账本

原 v4 的训练入口曾尝试一次，因注册文件白名单遗漏在第一次 fit 调用前停止：0 模型拟合、0 更新。用户随后明确批准一次恢复，沿用原最多 3 模型、10,584 次更新的未消耗预算；没有追加预算或自动重试。恢复执行中，种子 42、1019、2026 各完成且尝试 3,528 次更新、各有 12 次检查，共 3 次拟合、10,584 次更新；旧控制重训 0 次。每种子正中心呈现 56,448 次、负中心呈现 846,720 次。

训练和来源选择均已结束；没有调用开发冻结或验收入口，且未发现 future_identity、predictions、evaluation 工件。来源拒绝因此保持 control_guard95，默认模型未改变。该结论由执行路径、日志和工件状态支持，进程内隔离并非操作系统级审计，不能仅凭这些证据证明所有可能的外部读取均未发生。最终 House8/9/20 保持隔离。

## 核验与限制

独立审计复核了 135 个模型工件、三种子的完整批次索引及逐中心曝光、原初始化和 12 次检查；另据已保存的来源分数独立重算六块混淆矩阵和固定阈值门控，没有重新运行候选推理。执行阶段的专用只读 verify 和 report 各运行两次，重复输出一致；它们核验训练账本及来源决定，不检查开发结果。

早期准备中曾解析旧交付清单，无法排除其中包含留出户历史哈希元数据；未打开留出户数据或预测文件，也未计算其实际文件哈希。完整细节见独立复核报告和此前恢复准备报告。

## 证据

- 注册：`outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1/registration.json`。
- 训练和来源完成回执：同目录 `training_complete.json`、`source_selection.json`、`source_complete.json`。
- 进程记录与日志：`outputs/accuracy/microwave-task24-recovery-execution-20261028-v1/`。
- 独立复核：`outputs/accuracy/microwave-task24-independent-review-20261027-v1/Task24执行独立复核报告_20261028.md`。
- 两次只读执行核验：`outputs/accuracy/microwave-task24-execution-readonly-proof-20261028-v1/`。
- 首次中断：`outputs/accuracy/microwave-task24-interruption-audit-20261027-v1/receipt.json`。
'''
    ledger = {'prior_entry_attempts': 1, 'prior_actual_fits': 0, 'prior_updates': 0,
        'recovery_authorized_once': True, 'recovery_fits': 3, 'recovery_updates': 10584,
        'control_retrains': 0, 'automatic_retries': 0, 'seeds': summary['ledger'],
        'source_selected': 'control_guard95', 'development_stage_called': False}
    DEST.mkdir(parents=True)
    (DEST / '执行报告.md').write_text(report, encoding='utf-8')
    (DEST / 'attempt-ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    receipt = {'status': 'completed_source_rejected_control_retained',
        'registration_sha256': sha(RUN / 'registration.json'), 'authorization_sha256': sha(AUTH),
        'training_complete_sha256': sha(RUN / 'training_complete.json'),
        'source_selection_sha256': sha(RUN / 'source_selection.json'), 'source_complete_sha256': sha(RUN / 'source_complete.json'),
        'prior_interruption_sha256': sha(INTERRUPTION), 'readonly_proof_sha256': sha(PROOF / 'receipt.json'),
        'training_review_sha256': sha(REVIEW / 'training_review_20261028.receipt.json'),
        'source_review_sha256': sha(REVIEW / 'source_review_20261028.receipt.json'),
        'independent_report_sha256': sha(REVIEW / 'Task24执行独立复核报告_20261028.md'),
        'log_sha256': {p.name: sha(p) for p in logs},
        'files': {n: sha(DEST / n) for n in ('执行报告.md', 'attempt-ledger.json')}}
    (DEST / 'delivery_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
