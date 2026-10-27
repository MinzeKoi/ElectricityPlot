"""Finalize reviewed Task24 recovery preparation without authorizing execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1'
PROOF = ROOT / 'outputs/accuracy/microwave-task24-recovery-readonly-proof-20261027-v1'
AUDIT = ROOT / 'outputs/accuracy/microwave-task24-interruption-audit-20261027-v1'
REVIEW = ROOT / 'outputs/accuracy/microwave-task24-independent-review-20261027-v1'
PRIOR = ROOT / 'outputs/accuracy/microwave-task24-preparation-delivery-20261027-v1/human-budget-20261027.json'
DEST = ROOT / 'outputs/accuracy/microwave-task24-recovery-delivery-20261027-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert not DEST.exists(), 'A delivery is never overwritten'
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    digest = sha(RUN / 'registration.json')
    assert digest == '6d6754ee5e97ad69a619c107a2441b759048299cb2785214af01ada44b4961f7'
    assert sha(PRIOR) == '29e05d88c67d150354abf62067fd472ee5b9b9ad600ed5e66567d830c249f109'
    reg = read(RUN / 'registration.json')
    assert not reg['authorized'] and reg['maximum_new_fits'] == 3 and reg['maximum_updates'] == 10584
    interrupted = read(AUDIT / 'receipt.json')
    assert interrupted['actual_fits'] == interrupted['optimizer_updates'] == interrupted['backward_calls'] == 0
    assert interrupted['training_started_marker'] and not interrupted['models_directory']
    reviews = {}
    for name in ('independent_review_1.json', 'independent_review_2.json'):
        item = read(RUN / name)
        assert item['passed'] is True and item['reviewed_registration_sha256'] == digest
        reviews[name] = sha(RUN / name)
    proof = read(PROOF / 'receipt.json')
    assert proof['registration_sha256'] == digest and proof['verify_passes'] == proof['report_passes'] == 2
    assert proof['no_budget_entry_rejections'] == 4
    for item in proof['files']:
        assert sha(PROOF / item['name']) == item['sha256']
    diff = read(PROOF / 'source-diff.json')
    assert diff['new_registration_sha256'] == digest and diff['adapter_exactly_two_whitelist_additions']
    assert diff['package_python_files_equal'] == 106
    review_report = REVIEW / '恢复准备复核报告.md'
    assert review_report.is_file()
    DEST.mkdir(parents=True)
    template = {'authorized': False, 'approval_origin': None, 'experiment': 'microwave_task24_multihouse_v1',
        'registration_sha256': digest, 'review_sha256': reviews, 'additional_new_fits': 3,
        'maximum_updates': 10584, 'updates_per_seed': 3528, 'seeds': [42, 1019, 2026],
        'control_retrains': 0, 'automatic_retry': False, 'human_authorization_quote': '',
        'human_authorization_reference': '', 'status': 'not_authorized',
        'budget_note': 'The original approved maximum remains unspent; this is a one-time recovery request, not an additional budget.',
        'prior_budget_sha256': sha(PRIOR)}
    (DEST / 'recovery-budget-template.disabled.json').write_text(json.dumps(template, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = f'''# Task24 中断审计与恢复准备报告

恢复准备及独立复核已完成；**尚未授权重启训练**。原预算为最多 3 个候选模型、合计 10,584 次更新，每个种子 3,528 次；本次恢复请求沿用这笔尚未消耗的预算，不另加 3 个模型或增加更新上限。

2026-10-27 的真实执行仅尝试进入训练入口一次，随后在第一次调用 fit_one 前中止。原因是 v4 的训练隔离白名单遗漏了注册文件自身，计算其身份时被拒绝。`training_started.json` 已写入，模型目录和完成标记均不存在。独立审计确认实际模型拟合 0 次、优化器更新 0 次、反传 0 次、候选预测 0 次。进程已退出，原 v4、原授权回执、启动标记和日志保留。

新版本仅将 `registration.json` 加入训练和来源选择两个阶段的精确读取白名单；106 个来源包 Python 文件与 v4 完全相同。训练池、原归一化、16 通道模型、每批 16 正/240 普通负、12 次检查及其余协议不变。来源检查仍先执行既定 95% 保护与资格规则，再按平均精度、平均 F1 排序，完全平局保留控制；来源失败不得读取开发户，最终留出户保持隔离。控制模型不重训，任何再次中断不自动重试。

此前的合成测试在工作区外的系统临时目录执行，未覆盖工作区内注册文件实际隔离路径，因此未暴露本次遗漏。新测试在真实工作区内启用真实进程隔离，以合成数据走到替身首次 fit 调用并中止；来源选择的合成成功和保留控制路径也在真实隔离下通过。恢复快照的 18 项非训练合成测试、两次只读 verify/report 及独立复核通过；这些测试不代表真实模型已经训练，也不证明真实开发预测。

恢复注册 SHA-256：`{digest}`。原人类预算回执绑定 v4 注册，不可用于修改后的注册；因为原计划禁止自动重试，必须在本报告和复核证据基础上取得人类对这一次恢复的明确授权。`recovery-budget-template.disabled.json` 的 authorized 为 false，只是待确认模板。

同一独立审查者完成故障与代码差异审计、以及注册/测试/两次只读核验；两份回执在两次目录清单比较完成后新增，并绑定恢复注册，不代表两名审查人。只读属性及进程内路径白名单并非操作系统安全边界。

## 证据位置

- 原入口故障和进程记录：`outputs/accuracy/microwave-task24-execution-20261027-v1/`。
- 中断审计回执：`outputs/accuracy/microwave-task24-interruption-audit-20261027-v1/receipt.json`。
- 新注册及审查回执：`outputs/accuracy/microwave-task24-recovery-preparation-20261027-v1/`。
- 两次只读核验和源代码差异：`outputs/accuracy/microwave-task24-recovery-readonly-proof-20261027-v1/`。
- 独立复核报告：`outputs/accuracy/microwave-task24-independent-review-20261027-v1/恢复准备复核报告.md`。

此前准备中曾解析旧交付清单，无法排除其中包含留出户历史哈希元数据；未打开留出户数据或预测文件，也未计算其实际文件哈希。
'''
    (DEST / '恢复准备报告.md').write_text(report, encoding='utf-8')
    receipt = {'status': 'recovery_prepared_reviewed_not_authorized', 'registration_sha256': digest,
        'prior_budget_sha256': sha(PRIOR), 'interruption_sha256': sha(AUDIT / 'receipt.json'),
        'reviews': reviews, 'independent_report_sha256': sha(review_report),
        'readonly_proof_sha256': sha(PROOF / 'receipt.json'), 'source_diff_sha256': sha(PROOF / 'source-diff.json'),
        'files': {n: sha(DEST / n) for n in ('恢复准备报告.md', 'recovery-budget-template.disabled.json')},
        'prior_entry_attempts': 1, 'actual_fits': 0, 'updates': 0, 'recovery_authorized': False}
    (DEST / 'delivery_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
