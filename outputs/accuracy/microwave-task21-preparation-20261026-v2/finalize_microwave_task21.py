"""Seal Task21 preparation after independent replay; never execute training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_task21 as t


def child(args):
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([str(t.ROOT/'.venv-m2/Scripts/python.exe'),'-B',*map(str,args)],
        cwd=t.ROOT,env=env,capture_output=True,encoding='utf-8',errors='strict')
    # Both may be None if a caller changes capture policy; never concatenate None.
    out=(p.stdout or '')+(p.stderr or '')
    if p.returncode:raise RuntimeError(out)
    return out


def main(run):
    run=run.resolve()
    t.require(not (run/'preparation_manifest.json').exists(),'Completed preparation cannot be overwritten')
    before=t.snapshot(read_json(run/'inputs_before.json'))
    live=child(['-m','unittest','discover','-s','tests','-p','test_microwave_task21.py','-v'])
    snap=child(['-m','unittest','discover','-s',run,'-p','test_microwave_task21.py','-v'])
    with (run/'synthetic_live_tests.log').open('x',encoding='utf-8') as f:f.write(live)
    with (run/'synthetic_snapshot_tests.log').open('x',encoding='utf-8') as f:f.write(snap)
    result=t.verify(run)
    t.fresh(run/'independent_review.json',result)
    t.fresh(run/'authorization_template.disabled.json',dict(authorized=False,approval_origin='human',
        experiment='microwave_task21_hard_negative_v1',registration_sha256=t.sha256(run/'registration.json'),
        additional_new_fits=3,maximum_updates=10584,seeds=t.SEEDS,
        human_authorization_quote='',human_authorization_reference='',
        warning='Preparation authorization and all old receipts are insufficient. New explicit user budget required.'))
    proof=read_json(run/'sampling_proof.json')
    lines=['# Task21 实施准备报告','',
        '状态：实施准备与独立复核完成；默认关闭，等待绑定本次注册哈希的新训练预算。',
        '本轮新增真实训练 **0**，优化器构建 **0**，更新 **0**；仅为初始化身份核验构建网络，禁止优化器、反传和梯度计算。',
        '',f"注册 SHA256：`{t.sha256(run/'registration.json')}`",'',
        '| seed | H2/H19/H5 困难负窗 | 子段 | 困难曝光 | 批内重复率（整批） | 最大中心曝光占比 |',
        '|---|---|---:|---:|---:|---:|']
    for seed in t.SEEDS:
        p=proof[str(seed)]
        lines.append(f"| {seed} | {' / '.join(map(str,p['hard_counts']))} | {p['subsegments']} | {p['hard_presentations']} | {p['duplicate_rate']:.6%} | {p['hard_top_center_share']:.6%} |")
    lines += ['',
        '每种子全部3528批已封存。16正槽位及前120个普通负槽位身份、位置、顺序exact；后120个负槽位来自固定困难负集合。未重新打乱批次，额外RNG不触碰旧采样流。每种子困难负曝光423360、普通负曝光423360、正曝光56448。',
        '原336704窗/2275正例、原归一化、Task10与Task12同种子初始化、原事件采样及原干运行曝光已逐项重放。困难窗数量与提案完全一致，阈值未调整。House5角色按原prefix block0/1/2/3保留；片段按家庭/角色/UTC日期和8秒连续困难负切分，再按31中心切子段。',
        '独立实现使用分组游程边界和deque队列，核对全部替换索引、子段、逐中心曝光及家庭/日期/子段实际和期望曝光。期望曝光是该层级均匀轮转机制的长期期望，有限干运行的整数曝光不要求逐中心等于期望；完整差异可在exposure与exposure_groups查看。',
        '精确逐文件白名单、输入hash/mtime、源码快照、原始控制数值、每seed困难窗清单、子段CSV、3528×256完整索引、保留槽位及曝光证据均位于本目录。原数据不复制或改写。',
        'Task19仅使用control各seed train缓存的原336704前缀；完整压缩文件和身份文件hash绑定。NPZ压缩成员需整体解压再截前缀，后缀不参与数值核验、选择、采样或任何统计；不读取expanded输入/模型/缓存、Task19成绩报告或Task20错误查询。未读取或计算House17及最终8/9/20数据的数值、shape或hash。',
        '合成测试使用fake训练引擎和生成的静态数组覆盖授权、错行/错索引、缺失、阈值并列、空池/单段、边界、公开普通推理门控/缺失支持、来源拒绝0devread、训练/来源/封存/验收适配器、九格矩阵及八门槛。没有真实fit测试。',
        '未来执行入口已实现train→source_select→freeze→evaluate：新回执必须绑定本注册SHA、3fit/10584更新和种子；缺回执在模型训练、优化器和数据加载前拒绝。训练直接复用原fit_one(sealed indices)、Adam .001/unitBCE/原85检查点规则；来源固定L95/H95/r15，唯一对照Task14 control_guard95；来源未过则0开发读取。',
        '来源每seed每块R≥.95、无全开全关、meanF1不低于控制、每seed meanR下降≤.05、至少两seed P提高。通过才封存唯一候选的A/W50/候选三seed九格矩阵，核对公开event_gate.ordinary与固定配方，再一次原开发验收。开发沿用原八门槛、137523有效点、A原预测和其他电器状态，禁止后改选或追加训练。',
        '训练后verify/report自动追加实际训练批次日志/更新/曝光/初始化、原检查点规则、来源标量计数、封存九格矩阵、独立时域max与开发标量混淆矩阵及八门槛复核；它们保持只读。执行就绪仅经合成路由验证，未以真实训练或开发读取冒充验证。',
        '训练失败禁止重试；空困难池不新fit，保留原控制；不完整三seed矩阵拒绝开发。House17仍是反复使用的工程开发集，不能称独立测试；本分类试验不推断功率、运行段MAE或电量改善。',
        '新准备目录以排他创建，完成manifest不可覆盖；verify/report默认重复只读。最终另存只读回执，记录两遍live与snapshot verify/report前后所有输入和交付hash、mtime不变。',
        '', '下一步须由用户另行明确批准绑定上述注册SHA的最多3个新模型、10584更新；本轮准备授权不能作为训练预算。','']
    with (run/'PREPARATION_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
    t.require(t.snapshot(before)==before,'Finalization changed an input')
    t.fresh(run/'preparation_manifest.json',dict(status='prepared_verified_waiting_new_budget',
        registration_sha256=t.sha256(run/'registration.json'),new_fits=0,optimizer_updates=0,
        artifacts=t.inventory(run),inputs=before))
    delivery=t.snapshot([*before,*[run/k for k in t.inventory(run)]])
    calls=[]
    for iteration in (1,2):
        for script in (t.ROOT/'scripts/run_microwave_task21.py',run/'run_microwave_task21.py'):
            for command in ('verify','report'):
                out=child([script,command,'--run',run])
                calls.append(dict(iteration=iteration,script=str(script),command=command,stdout_sha256=__import__('hashlib').sha256(out.encode()).hexdigest()))
        t.require(t.snapshot(delivery)==delivery,'verify/report mutated input or delivery hash/mtime')
    receipt=run.with_name(run.name+'.readonly-receipt.json')
    t.fresh(receipt,dict(passed=True,passes=2,calls=calls,all_input_delivery_hash_mtime_unchanged=True,
        preparation_manifest_sha256=t.sha256(run/'preparation_manifest.json'),snapshots=delivery,new_fits=0,optimizer_updates=0))
    print(json.dumps(dict(status='prepared_sealed',registration_sha256=t.sha256(run/'registration.json'),
        manifest_sha256=t.sha256(run/'preparation_manifest.json'),readonly_receipt=str(receipt),new_fits=0,optimizer_updates=0),ensure_ascii=False,indent=2))


read_json=t.read_json
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
