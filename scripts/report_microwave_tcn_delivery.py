"""Summarize a completed frozen TCN run without opening any truth arrays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve(); root = Path(__file__).resolve().parents[1]
    done = read(run/"evaluation_complete.json")
    assert done["truth_csv_parses"] == 1
    source = read(run/"source_rows.json")
    metrics = read(run/"evaluation/metrics.json")
    summary = read(run/"evaluation/summary.json")
    training = read(run/"training_complete.json")
    assert training["fits"] == 6
    paired = []
    for seed in (42,1019,2026):
        a = next(r for r in metrics["rows"] if r["arm"] == "TCN31" and r["seed"] == seed)
        b = next(r for r in metrics["rows"] if r["arm"] == "TCN127" and r["seed"] == seed)
        sa = next(r for r in source["TCN31"] if r["seed"] == seed)
        sb = next(r for r in source["TCN127"] if r["seed"] == seed)
        with np.load(run/f"predictions/TCN31_{seed}.npz",allow_pickle=False) as aa, np.load(run/f"predictions/TCN127_{seed}.npz",allow_pickle=False) as bb:
            state_a, state_b = aa["state"], bb["state"]
            np.testing.assert_array_equal(aa["t"], bb["t"])
            np.testing.assert_array_equal(state_a >= 0, state_b >= 0)
            disagreement = int(((state_a >= 0) & (state_a != state_b)).sum())
        paired.append({"seed":seed,"difference_direction":"TCN127_minus_TCN31",
                       "source":{k:sb[k]-sa[k] for k in ("precision","recall","f1")},
                       "development":{k:b[k]-a[k] for k in ("precision","recall","f1","tp","fp","fn","tn")},
                       "state_disagreements_on_complete_core_centers":disagreement})
    times = {f"{a}/{s}": read(run/f"models/{a}/seed_{s}/model.json")["seconds"] for a in ("TCN31","TCN127") for s in (42,1019,2026)}
    benchmark_path = root/"outputs/accuracy/microwave-tcn-20261025-v2/benchmark.json"
    evidence = {"run":str(run),"status":"completed", "source_selection":read(run/"source_selection.json"),
                "source_rows":source,"development_rows":metrics["rows"],"summary":summary,"paired_context_differences":paired,
                "support":read(run/"context_support.json"),"recovery":read(run/"recovery_audit.json"),
                "successful_fits":6,"failed_source_fit_starts":1,"total_source_fit_starts":7,
                "successful_source_updates":21168,"failed_source_updates_inferred":294,"total_source_updates":21462,
                "source_train_seconds_by_model":times,"tests_passed":61,"new_tests_passed":21,
                "benchmark_reference":str(benchmark_path),"synthetic_benchmark":read(benchmark_path),
                "default_changed":False,"power_improvement_claim":False,"final_houses_accessed":False,
                "bindings":{name:digest(run/name) for name in ("registration.json","training_complete.json","source_rows.json","source_selection.json","freeze.json","inference_checks.json","evaluation_complete.json","evaluation/metrics.json","evaluation/summary.json","recovery_audit.json","report.json","report.md","preparation_verification.json","preparation_access.json","execution_authorization.json")}}
    evidence["bindings"]["benchmark_reference"] = digest(benchmark_path)
    evidence["bindings"]["tests"] = digest(root/"tmp/microwave-tcn-final-tests-20261025.log")
    evidence["bindings"]["v2_bug_reproduction"] = digest(root/"tmp/microwave-tcn-v2-keyerror-reproduction-20261025.log")
    evidence["bindings"]["delivery_script"] = digest(Path(__file__).resolve())
    evidence["independent_delivery_verification"] = read(run/"independent_delivery_verification.json")
    evidence["bindings"]["independent_delivery_verification.json"] = digest(run/"independent_delivery_verification.json")
    selected = summary["selected_before_development"]
    if selected is not None:
        candidate, baseline_mean = summary["means"][selected], summary["means"]["W50"]
        evidence["selected_gate_details"] = {"absolute_precision_at_least_008":candidate["precision"] >= .08,
             "mean_f1_gain":candidate["f1"]-baseline_mean["f1"],
             "fp_ratio_to_W50":candidate["fp"]/baseline_mean["fp"],
             "minimum_seed_recall":min(r["recall"] for r in metrics["rows"] if r["arm"]==selected)}
    out = root/"docs/MICROWAVE_TCN_RESULTS_20261025.json"
    out.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines = ["# 微波炉中心TCN与较长上下文：受控结果", "",f"实际运行：`{run}`。来源唯一预选：**{summary['selected_before_development']}**。开发验收：**{summary['development_acceptance_passed']}**。", "",
             "TCN31/TCN127各3seed，均5329参数、相同中心/标签/归一化/初始化与逐批索引，仅上下文长度不同。849正中心全部获得真实外圈，正中心平均新增91.87点。离线使用未来上下文。", "",
             "|模型|seed|TP|FP|FN|TN|P|R|F1|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metrics["rows"]:
        lines.append(f"|{r['arm']}|{r['seed']}|{r['tp']}|{r['fp']}|{r['fn']}|{r['tn']}|{r['precision']:.6f}|{r['recall']:.6f}|{r['f1']:.6f}|")
    lines += ["", "|模型|均P|均R|均F1|均FP|", "|---|---:|---:|---:|---:|"]
    for arm,r in summary["means"].items():
        lines.append(f"|{arm}|{r['precision']:.6f}|{r['recall']:.6f}|{r['f1']:.6f}|{r['fp']:.2f}|")
    lines += ["", "|seed|来源ΔP|来源ΔR|来源ΔF1|开发ΔP|开发ΔR|开发ΔF1|开发ΔFP|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in paired:
        a,b=r["source"],r["development"]
        lines.append(f"|{r['seed']}|{a['precision']:+.6f}|{a['recall']:+.6f}|{a['f1']:+.6f}|{b['precision']:+.6f}|{b['recall']:+.6f}|{b['f1']:+.6f}|{b['fp']:+d}|")
    lines += ["", "上表差值均为TCN127−TCN31。两个新TCN之间是上下文对照；新TCN与旧partial网络结构不同，差异不能全归因于上下文。开发结果不用于改选或追加训练。", "",
              "验收门槛：均P≥.08、每seed R≥.85、相对W50均F1增≥.02（1e−12边界容差）、至少2seed P/F1同升、各seed F1降≤.02、总FP降≥30%、非退化。", "",
              "v2首次训练因验证元数据缺window_samples失败，未保存可恢复首轮权重、0来源验证分数文件、0开发真值读取；294更新计1次失败启动。用户明确授权一次工程恢复。v3六个有效拟合21168更新；累计7次启动、21462更新，未扩候选或调参。新seed42初始化及前294批索引与失败运行完全相同。旧错误已由封存v2实际合成训练循环复现，修复覆盖31/127训练到验证及普通加载。补充全数组形状核验在v3启动后记录，双长度合成集成测试在v3准备前通过；不把事后核验表述为训练前完成。", "",
              "61项测试通过（21新增、40既有）。准备独立核验保留全部259806训练中心和109786验证中心。来源唯一选择与全部预测先冻结，普通推理一致、原W50共同支持每seed137523、水壶/洗衣机及原微波炉功率A逐点不变，之后一次解析House17真值并评分。最终House8/9/20未访问，默认未切换，无功率改善声明。", "",
              f"合成CPU基准（同一固定网络，v2修复后的内存测量）：31/127更新约9.67/17.43ms，六拟合纯训练预计4.78分钟，进程峰值约633MiB，另需验证/保存/校验。v3六模型实际训练加验证/保存合计{sum(times.values())/60:.2f}分钟。", "",
              "普通预测（输入/输出CSV路径按实际替换，输出须不存在）：", "", "```powershell",f'.venv-m2/Scripts/python.exe "{run.as_posix()}/predict_microwave_tcn.py" --model "{run.as_posix()}/models/TCN127/seed_42" --aggregate "aggregate.csv" --output "microwave-tcn-output.csv"', "```", "",
              "TCN31替换臂名即可；旧partial使用任务5的predict_microwave_mask.py及其partial_mask模型，不能用TCN加载器。完整证据与SHA256见同名JSON；运行内report由封存verify流程生成。"]
    (root/"docs/MICROWAVE_TCN_RESULTS_20261025.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"results":str(out),"selected":summary["selected_before_development"],"accepted":summary["development_acceptance_passed"],"means":summary["means"]},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
