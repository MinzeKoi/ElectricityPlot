"""Seal Task22 after full static pipeline and independent replay; zero real fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_task22 as t


def child(args):
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run([str(t.ROOT/'.venv-m2/Scripts/python.exe'),'-B',*map(str,args)],cwd=t.ROOT,
        env=env,capture_output=True,encoding='utf-8',errors='strict')
    out=(p.stdout or '')+(p.stderr or '')
    if p.returncode:raise RuntimeError(out)
    return out


def main(run):
    run=run.resolve()
    t.require(not (run/'preparation_manifest.json').exists(),'Completed preparation cannot be overwritten')
    before=t.snapshot(t.read_json(run/'inputs_before.json'))
    for label,folder in (('live',t.ROOT/'tests'),('snapshot',run)):
        source=(t.ROOT/'src') if label=='live' else run/'source'
        code='import sys;sys.path.insert(0,'+repr(str(source))+');from virtual_submeter.accuracy.microwave_task22_review import review_preparation;from virtual_submeter.accuracy.microwave_task22 import diagnostics;print("cold independent-review-first import passed")'
        cold=child(['-c',code])
        with (run/f'cold_import_{label}.log').open('x',encoding='utf-8') as f:f.write(cold)
        log=child(['-m','unittest','discover','-s',folder,'-p','test_microwave_task22.py','-v'])
        with (run/f'synthetic_{label}_tests.log').open('x',encoding='utf-8') as f:f.write(log)
    result=t.verify(run)
    t.fresh(run/'independent_review.json',result)
    registration=t.sha256(run/'registration.json')
    t.fresh(run/'authorization_template.disabled.json',dict(authorized=False,approval_origin='human',experiment=t.SCHEMA,
        registration_sha256=registration,additional_new_fits=3,maximum_updates=10584,seeds=t.SEEDS,
        human_authorization_quote='',human_authorization_reference='',warning='Only a NEW explicit human budget bound to this SHA can authorize execution. Preparation and Task21 receipts do not authorize training.'))
    lines=['# Task22 容量对照实施准备报告','',
        '状态：准备封存，默认关闭，等待新的绑定预算。真实fit 0；优化器构建0；更新0；反传0；House17与最终8/9/20数值、shape、hash读取0。',
        '',f'注册 SHA256：`{registration}`','',
        '唯一未来变量为TCN通道16→32，参数5329→19809。project、六个膨胀conv及head首层输入宽度变化；核3、膨胀1/2/4/8/16/32、残差公式、head隐藏32、中心读出不变。',
        '原训练336704窗/2275正、原House5prefix与block4/5验证34351/201、C127、原归一化/mask/200W、unitBCE/Adam .001/CPU4固定。三seed42/1019/2026每个3528步(12×294)，16正/240均匀负，原3528×256完整索引逐行重放exact。未使用Task21困难采样、Task18追加数据、背景增强、融合或新特征/损失。',
        '旧16通道三seed初始化与Task10/Task12历史逐张量exact；旧控制在原批边界来源普通预测逐点exact。新32通道按同规则/同seed可重复，19809参数，完整127训练mask前向与严格core31公开推理支持验证；新旧张量不声称相同，不裁剪或拼接旧权重。',
        '已实现显式channels接口，复用原fit_one训练引擎；旧缺省16及旧元数据行为保持。32通道模型元数据必须明确tcn_channels=32/parameter_count=19809，不能按缺省16加载。',
        '执行入口为本目录run_microwave_task22.py。训练、来源筛选、freeze、evaluate和诊断均先验证新的Task22注册SHA回执；缺失或Task21旧回执在训练数据、模型训练、优化器/反传前拒绝。真实执行强制从本次不可变source快照加载；数据/旧控制身份继续核对，不依赖未来工作区源文件不变。',
        '来源仍按原85保护、worstP/meanP/F1/早epoch选择完整36checkpoint；复核读取完整event及日期边界。L95/H95/r15与Task14 control_guard95固定；每seed每块R≥.95非退化、meanF1≥control、每seed meanR下降≤.05、至少2seed P提高。失败则0devread。',
        '通过后唯一候选wide32与A/W50三seed完整九格冻结，公开普通推理exact后，仅单次原137523有效点开发验收。八门槛为meanP≥.08、各seedR≥.85、meanF1比W50≥+.02、至少2seed P/F1同升、各seedF1下降≤.02、meanFP减少≥30%、非全开全关、支持与其他设备原状态exact。不得开发后改选、改宽、延训；最终8/9/20隔离、默认模型不切换。',
        '新增diagnostics入口在训练后可对固定选中模型及旧控制的原完整训练/来源支持与共同strict_core31两视图报告AP、FPR、固定点混淆；只解释、不调阈值或改选，来源失败仍不读开发。AP用同分分组的第二实现复核。',
        '完整合成测试构建独立静态训练支持及带event/跨UTC日期边界的来源数组，用fakefit写出3seed×12检查点、实际格式批日志/初始化/曝光。真实走train→source_select→freeze→evaluate→verify/report全部分支；verify/report本身未mock，包含两套checkpoint数值算法、L95/H95、独立单调deque时域最大、标量混淆与开发八门槛。静态九格有137523个生成点，不含真实House17。',
        '合成拒绝路径覆盖缺预算/旧预算、缺九格矩阵、低Recall/全关、重复评价、严格core31缺失、错误32宽度元数据；fixture全程禁止优化器/反传。这里只fakefit，不运行任何真实或合成优化训练。Task21工作区复核器遗漏event已修复，Task21 v2封存文件未修改。',
        '冷启动诊断补充发现旧v2在白名单内部延迟导入会探测未登记pyc，旧v3直接提前导入复核模块产生反向导入循环。v2/v3原文件及失败记录保留；本版只将独立AP放入无协议依赖的数值模块，并在进入限制前导入数值帮助函数；live/snapshot新进程复核模块优先导入验证通过。原诊断24格、12份缓存及口径不变。',
        '独立实现复核在本对话完成：原采样使用另一套deque消费与独立日期/事件分组；新旧初始化和模型结构单独重建，来源日期/事件隔离重查。未新增subagent；协调者可做外部核验。',
        '精确逐文件白名单、输入hash/mtime、源数据/控制/代码身份、完整批矩阵、新初始化、中文报告和测试日志均封存。future_identity只从既有身份元数据登记未来路径，未打开、hash或解析对应开发文件；不读取混有开发结果的旧报告。',
        '完成目录不可覆盖；已有完整prepare转只读verify。finalize外部只读回执记录两遍live与snapshot prepare/verify/report共12次及所有输入/交付hash和mtime不变。',
        '', '待预算：最多3个新模型、总10584更新仅提议；必须另行由用户批准并绑定本注册SHA。Task21预算已花完，不能复用。','']
    with (run/'PREPARATION_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
    t.require(t.snapshot(before)==before,'Finalization changed input')
    t.fresh(run/'preparation_manifest.json',dict(status='prepared_verified_waiting_new_budget',registration_sha256=registration,
        new_fits=0,optimizers=0,optimizer_updates=0,backward_calls=0,artifacts=t.inventory(run),inputs=before))
    delivery=t.snapshot([*before,*[run/k for k in t.inventory(run)]])
    calls=[]
    for iteration in (1,2):
        for script in (t.ROOT/'scripts/run_microwave_task22.py',run/'run_microwave_task22.py'):
            for command in ('prepare','verify','report'):
                out=child([script,command,'--run',run]);calls.append(dict(iteration=iteration,script=str(script),command=command,stdout_sha256=hashlib.sha256(out.encode()).hexdigest()))
        t.require(t.snapshot(delivery)==delivery,'Readonly routes changed input/delivery hash/mtime')
    receipt=run.with_name(run.name+'.readonly-receipt.json')
    t.fresh(receipt,dict(passed=True,passes=2,calls=calls,all_input_delivery_hash_mtime_unchanged=True,
        preparation_manifest_sha256=t.sha256(run/'preparation_manifest.json'),snapshots=delivery,new_fits=0,optimizers=0,optimizer_updates=0,backward_calls=0))
    print(json.dumps(dict(status='prepared_sealed',registration_sha256=registration,manifest_sha256=t.sha256(run/'preparation_manifest.json'),readonly_receipt=str(receipt),new_fits=0,optimizer_updates=0),ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);main(parser.parse_args().run)
