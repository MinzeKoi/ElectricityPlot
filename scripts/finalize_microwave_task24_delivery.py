"""Create the Task24 preparation delivery; no model execution or authorization."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/accuracy/microwave-task24-preparation-20261027-v4'
PROOF = ROOT / 'outputs/accuracy/microwave-task24-readonly-proof-20261027-v4'
INDEPENDENT = ROOT / 'outputs/accuracy/microwave-task24-independent-review-20261027-v1'
DEST = ROOT / 'outputs/accuracy/microwave-task24-preparation-delivery-20261027-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert not DEST.exists(), 'A delivery is never overwritten'
    read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    reg = read(RUN / 'registration.json')
    digest = sha(RUN / 'registration.json')
    assert digest == '5bc276ac7fce10d63be1033bee2072dee587c6253bdbaab71168b69579dbbe37'
    assert reg['authorized'] is False and reg['counts'] == {'centers': 1393867, 'positive': 4389}
    reviews = {}
    for name in ('independent_review_1.json', 'independent_review_2.json'):
        path = RUN / name
        item = read(path)
        assert item['passed'] is True and item['reviewed_registration_sha256'] == digest
        reviews[name] = sha(path)
    proof = read(PROOF / 'receipt.json')
    assert proof['registration_sha256'] == digest and proof['verify_passes'] == proof['report_passes'] == 2
    assert proof['no_budget_entry_rejections'] == 4
    for row in proof['files']:
        assert sha(PROOF / row['name']) == row['sha256']
    audit = read(INDEPENDENT / 'receipt.json')
    assert audit['index_exit'] == audit['windows_exit'] == 0
    report = (RUN / 'MICROWAVE_TASK24_PREPARATION_DRAFT_20261027.md').read_text(encoding='utf-8')
    report = report.replace('# Task24 实施准备草稿（未完成，不授权训练）', '# Task24 多家庭 C127 分类对照：实施准备报告')
    report = report.replace('## 边界和未完成事项', '## 边界与预算')
    report = report.replace('本草稿不构成实施准备完成或新增训练预算。源码和数据注册已建立，仍须完成独立复核与只读核验；不得以本草稿开始训练或读取开发、最终数据。',
        '实施准备及独立复核已完成。本报告不授权训练；新的候选模型预算仍须由用户单独确认。')
    report = report.replace('真实模型推理未在准备阶段执行；仍须独立审查注册与代码、完成两次只读核验和复核回执。本草稿不授权训练。',
        '真实候选模型推理未在准备阶段执行；最终注册经两轮只读核验和独立复核。本报告不授权训练。')
    report += f'''
\n## 固定训练计划及停止条件

拟议为种子 42、1019、2026 各训练一个 16 通道（5,329 参数）候选，复用原 C127、归一化、unit_BCE、Adam 和固定初始化；每种子 3,528 次更新，每 294 次检查一次，共 12 次检查。最多新增 3 个候选、10,584 次更新。直接复用三份旧控制模型，不重训控制；不自动重试、不追加候选、权重、采样或网络搜索。

来源阶段保留 House5 原验证后缀、原检查点选择及 95% 召回保护。来源未通过时保留 control_guard95，不读取 House17 开发；最终 House8/9/20 始终隔离。来源通过后才按既定顺序冻结候选和执行原开发验收的八项门槛；这些步骤均未在本轮执行。

## 复核与封存身份

最终注册 SHA-256：`{digest}`。对新旧来源索引、全部采样批次、逐中心曝光及新增 1,057,163 个窗口进行了独立复核；新增格点重算与窗口比对使用相同原始切片，并非独立数据源。Task23 历史审计比对的处理后文件本轮不可读。源码、真实输入身份与来源、预算及开发门控经复核；17 项合成非训练测试通过，四个执行入口缺预算时均拒绝。

同一独立审查者完成两轮复核，分别针对来源索引和窗口、以及协议和执行链；不声称两名审查者。其两份回执绑定上述注册。两轮只读 verify/report 的结果一致，目录清单对比完成后才有意新增两份复核回执。注册的 118 个已有文件设置了 Windows 只读属性；该属性和进程内精确路径白名单不是操作系统安全边界。

参考工件：`outputs/accuracy/microwave-task24-preparation-20261027-v4/registration.json`、同目录两份 `independent_review_*.json`、`outputs/accuracy/microwave-task24-readonly-proof-20261027-v4/receipt.json`、`outputs/accuracy/microwave-task24-independent-review-20261027-v1/receipt.json`。本交付中的 `budget-template.disabled.json` 仅为待用户确认的预算模板，不是授权回执。
'''
    template = {'authorized': False, 'approval_origin': None, 'experiment': 'microwave_task24_multihouse_v1',
        'registration_sha256': digest, 'review_sha256': reviews, 'additional_new_fits': 3,
        'maximum_updates': 10584, 'updates_per_seed': 3528, 'seeds': [42, 1019, 2026],
        'control_retrains': 0, 'automatic_retry': False, 'human_authorization_quote': '',
        'human_authorization_reference': '', 'status': 'not_authorized'}
    DEST.mkdir(parents=True)
    (DEST / '报告.md').write_text(report, encoding='utf-8')
    (DEST / 'budget-template.disabled.json').write_text(json.dumps(template, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    receipt = {'status': 'prepared_reviewed_budget_not_authorized', 'registration_sha256': digest,
        'reviews': reviews, 'readonly_proof_sha256': sha(PROOF / 'receipt.json'),
        'independent_data_proof_sha256': sha(INDEPENDENT / 'receipt.json'),
        'delivery_files': {name: sha(DEST / name) for name in ('报告.md', 'budget-template.disabled.json')},
        'actual_fits': 0, 'optimizer_updates': 0, 'candidate_predictions': 0,
        'additional_budget_authorized': False}
    (DEST / 'delivery_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
