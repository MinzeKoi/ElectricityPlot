"""Seal recovery preparation only after static fakefit and independent replays."""
import argparse,hashlib,json,os
from pathlib import Path
import subprocess,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source' if (HERE/'registration.json').exists() else HERE.parent/'src'))
from virtual_submeter.accuracy import microwave_task22_recovery as t


def child(args):
 env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 p=subprocess.run([str(t.ROOT/'.venv-m2/Scripts/python.exe'),'-B',*map(str,args)],cwd=t.ROOT,env=env,capture_output=True,encoding='utf-8',errors='strict')
 out=(p.stdout or '')+(p.stderr or '')
 if p.returncode:raise RuntimeError(out)
 return out


def main(run):
 run=run.resolve();t.require(not (run/'preparation_manifest.json').exists(),'Completed preparation cannot be overwritten')
 before=t.snapshot(t.read_json(run/'inputs_before.json'))
 for label,folder in (('live',t.ROOT/'tests'),('snapshot',run)):
  print('Static fakefit route tests: '+label,flush=True)
  log=child(['-m','unittest','discover','-s',folder,'-p','test_task22_recovery.py','-v'])
  with (run/f'synthetic_{label}_tests.log').open('x',encoding='utf-8') as f:f.write(log)
 print('Independent replay of original data, batches and two completed models',flush=True)
 independent=t.independent_verify(run)
 t.fresh(run/'independent_review.json',independent)
 reg=t.sha256(run/'registration.json');identity=t.read_json(run/'interruption_identity.json')
 t.fresh(run/'authorization_template.disabled.json',dict(authorized=False,approval_origin='human',experiment=t.SCHEMA,
  registration_sha256=reg,original_registration_sha256=t.ORIGINAL_SHA,interruption_audit_sha256=t.AUDIT_SHA,
  reused_model_identity_sha256=t.sha256(run/'reused_model_identity.json'),failed_model_inventory_sha256=identity['failed_model_inventory_sha256'],
  seed=2026,additional_new_fits=1,maximum_updates=3528,fresh_from_original_initial=True,
  human_authorization_quote='',human_authorization_reference='',warning='A separate explicit human recovery authorization is required. Original Task22 budget does not authorize retry.'))
 text='''# Task22 seed2026 中断恢复准备报告

状态：恢复工程与静态路径完成并封存，默认关闭；等待绑定本注册的独立1fit/3528新预算。此次准备0真实或合成训练、0优化器、0更新、0反传、0开发/最终读取。

恢复注册SHA256：`'''+reg+'''`

原v4注册SHA256：`'''+t.ORIGINAL_SHA+'''`

原v4两份完整模型 seed42/1019 各3528完成，批日志、初始化、逐中心曝光、12检查点和原85预选均经中断审计及本次独立复核。seed2026已完成567、已尝试568；第568次是否实际更新未知，已保存检查点无Adam状态。原中断目录、日志、源码、索引、权重、注册和预算全部保留，不调用旧train入口或声称精确续训。

累计账本：原3次fit尝试中，2完整、1中断；完成更新下界7623，尝试上界7624。若获得新预算并完整重跑，实际fit尝试共4次，累计完成更新下界11151、尝试上界11152；有效候选为2个复用+1个恢复模型，3seed各3528，有效矩阵10584更新。本次新fit始终计1，旧完整模型复用计2，历史失败计1，不能把总消耗说成10584。开发验收summary只复用原门槛计算；fit及更新统计来自验证过的恢复账本，显式记录actual_new_fits=1等字段，不保留旧框架含糊的new_fits=3/updates=10584。独立复核拒绝错误或遗漏历史消耗的summary。

拟恢复seed2026严格从原initial32和原全部3528×256批索引重新训练，16正/240均匀负、unitBCE/Adam .001、C127、TCN32/19809参数、12×294步、CPU4；无新增数据/采样/特征/损失或新阈值。恢复失败不重试。入口显式设置并报告实际CPU线程4，UTF-8和禁用字节码；旧两个模型逐文件复制并检查hash，保持原model.json绑定原v4注册；恢复模型绑定本注册。源码、引擎和入口从本目录不可变快照加载，原v4源/工件不被改写。

独立人类回执须同时绑定本恢复SHA、原v4 SHA、中断审计manifest SHA、复用模型身份文件SHA、失败目录清单SHA、seed2026，明确仅1fit和3528更新并从原初始化重跑。缺回执或旧预算在读取训练数据、训练模型、创建优化器及fit前拒绝；disabled模板不是授权。

共用来源/冻结/验收算法以显式恢复上下文核验实际账本，不伪称新fit=3：完整有效矩阵和双注册身份先通过后，沿原85检查点与固定Task14 control_guard95/L95/H95/r15资格；任一来源门槛失败即0开发读取。通过才冻结唯一候选及A/W50/候选九格，公开推理exact后一次原137523点八门槛验收；禁止开发后改选、改宽度、延训。最终8/9/20隔离、默认不切换。固定诊断保留24格、12缓存、同窗AP与固定点统计，不改选。

完整静态夹具经fakefit走恢复1fit/两个复用→source→diagnostics→freeze→evaluate→实际verify/report；真实执行时复核会用固定模型重放12份诊断缓存并独立复算24格AP/混淆，包含跨日期的完整event支持、缺预算/旧回执拒绝、旧工件篡改和缺矩阵拒绝、低Recall/全关时source拒绝且0开发读取、重复训练/诊断/评价拒绝。fixture仅构造网络并使用静态输出，不做任何优化训练或反传。独立复核另以deque重放原3seed全部批索引、曝光和32初始化，重查两份旧模型各12检查点及日期/事件；未新增subagent。

新目录排他创建，失败尝试保留；完成后prepare/verify/report/independent_verify默认只读。两遍工作区与快照检查全部输入及交付hash/mtime，外部只读回执提供证据。
'''
 with (run/'PREPARATION_REPORT.zh.md').open('x',encoding='utf-8',newline='\n') as f:f.write(text)
 t.require(t.snapshot(before)==before,'Preparation input changed')
 t.fresh(run/'preparation_manifest.json',dict(status='prepared_waiting_new_one_fit_budget',registration_sha256=reg,new_fits=0,optimizers=0,optimizer_updates=0,backward_calls=0,artifacts=t.inventory(run),inputs=before))
 allfiles=t.snapshot([*before,*[run/k for k in t.inventory(run)]]);calls=[]
 for iteration in (1,2):
  print(f'Read-only live/snapshot pass {iteration}',flush=True)
  for script in (t.ROOT/'scripts/run_task22_recovery.py',run/'run_task22_recovery.py'):
   for command in ('prepare','verify','report','independent_verify'):
    out=child([script,command,'--run',run]);calls.append(dict(iteration=iteration,script=str(script),command=command,stdout_sha256=hashlib.sha256(out.encode()).hexdigest()))
  t.require(t.snapshot(allfiles)==allfiles,'Read-only call changed input/delivery hash/mtime')
 receipt=run.with_name(run.name+'.readonly-receipt.json')
 t.fresh(receipt,dict(passed=True,passes=2,calls=calls,all_input_delivery_hash_mtime_unchanged=True,preparation_manifest_sha256=t.sha256(run/'preparation_manifest.json'),snapshots=allfiles,new_fits=0,optimizer_updates=0))
 print(json.dumps(dict(status='recovery_prepared_sealed',registration_sha256=reg,manifest_sha256=t.sha256(run/'preparation_manifest.json'),readonly_receipt=str(receipt),new_fits=0,optimizer_updates=0),ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
