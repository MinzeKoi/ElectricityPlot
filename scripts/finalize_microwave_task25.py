"""Deliver reviewed Task25 preparation and a disabled budget template."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/accuracy/microwave-task25-preparation-20261028-v2'
PROOF=ROOT/'outputs/accuracy/microwave-task25-readonly-proof-20261028-v2'
REVIEW=ROOT/'outputs/accuracy/microwave-task25-independent-review-20261028-v1'
DEST=ROOT/'outputs/accuracy/microwave-task25-preparation-delivery-20261028-v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
    return h.hexdigest()


def main():
    assert not DEST.exists()
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    reg=read(RUN/'registration.json');digest=sha(RUN/'registration.json')
    assert digest=='a59008deaf622bb139321e41a81cfb47f2848f4c82db7f71a4ac117240f6dadc'
    assert reg['authorized'] is False and reg['maximum_new_fits']==3 and reg['maximum_updates']==10584
    reviews={}
    for name in ('independent_review_1.json','independent_review_2.json'):
        item=read(RUN/name);assert item['passed'] and item['reviewed_registration_sha256']==digest
        reviews[name]=sha(RUN/name)
    proof=read(PROOF/'receipt.json')
    assert proof['registration_sha256']==digest and proof['verify_passes']==proof['report_passes']==2 and proof['no_budget_entry_rejections']==4
    for item in proof['files']:assert sha(PROOF/item['name'])==item['sha256']
    sampling=read(REVIEW/'sampling_receipt.json')
    assert sampling['exit_code']==0
    assert not any((RUN/n).exists() for n in ('training_started.json','models','source_selection.json','future_identity.json','predictions','evaluation'))
    report='''# Task25 正例来源配额对照：实施准备报告

准备和独立复核已完成；**新增训练预算尚未授权**。Task25 只检验一个预定工程方案：在 Task24 相同七户池和固定更新量下，将原三户正例呈现份额从约三分之一调整为三分之二。2/3 是查看 Task24 来源结果后指定的单一方案，并非比例搜索结果；它同时减少新增户曝光，结果不能证明单一因素的纯因果作用。

| 项目 | 每个种子 |
|---|---:|
| 来源池中心 / 开机中心 | 1,393,867 / 4,389 |
| 固定更新 / 检查点 | 3,528 / 12 |
| 原三户 / 新四户正例呈现 | 37,632 / 18,816 |
| 负例呈现 | 846,720 |
| 每批正 / 负 | 16 / 240 |

每三批旧组三批配额依次 11、11、10；新组相应 5、5、6。每 294 步检查区间恰含 98 个完整周期。240 个负例的索引、位置和顺序逐批与 Task24 一致。组内使用原有 `(家庭, UTC 日) → 该日片段 → 片段中心` 的循环规则；各组独立固定随机流，正槽顺序也由单独固定流打散。三种子各不重复中心分别为 851,029、851,029、851,028；重复曝光不等于独立物理使用次数。

| 家庭 | 正例呈现：42 / 1019 / 2026 |
|---|---:|
| House2 | 13,353 / 13,354 / 13,353 |
| House19 | 14,567 / 14,566 / 14,568 |
| House5 前缀 | 9,712 / 9,712 / 9,711 |
| House10 | 6,982 / 6,982 / 6,980 |
| House12 | 3,946 / 3,943 / 3,948 |
| House15 | 4,853 / 4,856 / 4,853 |
| House18 | 3,035 / 3,035 / 3,035 |

逐户、UTC日、片段和逐中心曝光及三种子完整批次保存在封存准备目录；日字段为自 1970-01-01 起的 UTC 日数。数据池、C127 掩码、归一化、16 通道模型、初始化、unit_BCE 和 Adam 均复用 Task24 身份。旧控制和 Task24 三个模型不重训。Task24 是冻结过程对照；来源晋升仍以 control_guard95 的既定逐块 95% 召回、非退化、相对资格和平均精度/平均 F1 排序为准，完全平局保留控制。仅优于 Task24 不足以晋升；来源不通过不读取 House17，最终 House8/9/20 始终隔离。

本轮零真实拟合、零更新、零反传、零候选推理，未读取开发或最终数据，未下载或修改旧数据池与默认模型。19 项非训练测试包括真实工作区和真实隔离下到替身首个训练调用的测试、来源选择与拒绝路径，以及禁止复用 Task24 预算的测试。独立审查逐批重放采样、核对全部负槽及曝光；两次只读 verify/report 输出一致。只读属性与进程内路径控制不是操作系统安全边界。

新增四户窗口沿用 Task24 已审计并由原始切片重建的封存工件；处理后四户文件当时不可读，不能声称本轮再次直接核验。Task24 早期核对曾解析旧交付清单，无法排除其中含留出户历史哈希元数据；本轮未打开留出户数据或预测文件，也未计算其实际文件哈希。

拟议上限为最多 3 个新增候选、合计 10,584 次更新，每种子 3,528 次。该预算需另行由人类授权；Task24 已完成的三次拟合不属于本次预算，也不能重用其授权。不得自动重试或搜索其他配额。

封存注册、批次和曝光：`outputs/accuracy/microwave-task25-preparation-20261028-v2/`；预注册：`docs/MICROWAVE_TASK25_PROPOSAL_20261028.md`；只读证据：`outputs/accuracy/microwave-task25-readonly-proof-20261028-v2/`；独立复核：`outputs/accuracy/microwave-task25-independent-review-20261028-v1/`。同一独立审查者完成两轮复核；回执在只读目录清单比较之后新增，不代表两名审查人。
'''
    template={'authorized':False,'approval_origin':None,'experiment':'microwave_task25_positive_share_v1','registration_sha256':digest,
        'review_sha256':reviews,'additional_new_fits':3,'maximum_updates':10584,'updates_per_seed':3528,'seeds':[42,1019,2026],
        'control_retrains':0,'task24_retrains':0,'automatic_retry':False,'human_authorization_quote':'',
        'human_authorization_reference':'','status':'not_authorized'}
    DEST.mkdir(parents=True)
    (DEST/'准备报告.md').write_text(report,encoding='utf-8')
    (DEST/'budget-template.disabled.json').write_text(json.dumps(template,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    receipt={'status':'prepared_reviewed_budget_not_authorized','registration_sha256':digest,'reviews':reviews,
        'sampling_review_sha256':sha(REVIEW/'sampling_receipt.json'),'readonly_proof_sha256':sha(PROOF/'receipt.json'),
        'files':{n:sha(DEST/n) for n in ('准备报告.md','budget-template.disabled.json')},
        'actual_fits':0,'updates':0,'candidate_predictions':0,'budget_authorized':False}
    (DEST/'delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
