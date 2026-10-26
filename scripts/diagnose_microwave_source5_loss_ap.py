"""Post-freeze exact tied-score AP diagnosis; no fitting or threshold selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(path,keys):
    with np.load(path,allow_pickle=False) as z:
        return {key:z[key] for key in keys}


def grouped_ap_numpy(scores,y):
    order = np.argsort(-scores,kind='stable')
    p,yy = scores[order],y[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1]
    tp = np.cumsum(yy)[ends]
    precision = tp/(ends+1)
    positive_in_group = np.diff(np.r_[0,tp])
    return float(np.sum(positive_in_group*precision)/int(y.sum()))


def grouped_ap_scalar(scores,y):
    groups = {}
    for p,positive in zip(scores,y):
        key = float(p)
        on,n = groups.get(key,(0,0))
        groups[key] = (on+int(positive),n+1)
    tp,total,weighted = 0,0,0.
    for key in sorted(groups,reverse=True):
        positives,n = groups[key]
        tp,total = tp+positives,total+n
        weighted += positives*(tp/total)
    return weighted/int(y.sum())


def diagnose(run,output):
    assert not output.exists(),'New diagnostic output required'
    manifest = json.loads((run/'final_delivery_manifest.json').read_text(encoding='utf-8'))
    assert manifest['development_acceptance_passed'] is False and manifest['source_selected']=='new_gate'
    bound = {}
    def load(name,keys):
        path = run/name
        assert digest(path)==manifest['artifacts'][name],name
        bound[name] = manifest['artifacts'][name]
        return fields(path,keys)
    truth = load('evaluation/truth_once.npz',['t','y'])
    rows = []
    for seed in (42,1019,2026):
        w = load(f'predictions/W50_{seed}.npz',['t','raw'])
        np.testing.assert_array_equal(w['t'],truth['t'])
        mask = np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1])
        y = truth['y'][mask]>=200
        assert int(mask.sum())==137523 and int(y.sum())==383
        row = dict(seed=seed,support=137523,positives=383)
        for family in ('old_point','new_point'):
            z = load(f'predictions/{family}_{seed}.npz',['t','scores'])
            np.testing.assert_array_equal(z['t'],truth['t'])
            p = z['scores'][mask]
            assert np.isfinite(p).all()
            ap_numpy,ap_scalar = grouped_ap_numpy(p,y),grouped_ap_scalar(p,y)
            assert abs(ap_numpy-ap_scalar)<1e-12
            row[family+'_ap'] = ap_numpy
        row['delta_ap'] = row['new_point_ap']-row['old_point_ap']
        rows.append(row)
    result = dict(schema='post_freeze_tied_AP_diagnosis_v1',diagnostic_only=True,
        parent_run=str(run.resolve()),parent_manifest_sha256=digest(run/'final_delivery_manifest.json'),
        parent_artifacts=bound,rows=rows,new_fits=0,updates=0,thresholds_selected=0,
        source_selection_changed=False,formal_development_acceptance_passed=False,
        tied_score_groups_exact=True,two_independent_AP_implementations_agree=True,
        interpretation='All three seeds improve score ranking on repeatedly used House17; this is not independent generalization and does not override seed1019 gate recall failure.')
    output.mkdir(parents=True)
    (output/'diagnostic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# 冻结后 AP 诊断附录','',
        '仅诊断：没有拟合、选阈值、选候选或修改来源预选。原交付保持封存，正式结论仍为未达标。',
        '全部使用原W50同一137523点、383正点；同分数作为完整并列组计算精确Average Precision（非PR曲线梯形面积）。向量与独立标量分组实现误差小于1e-12。','',
        '| seed | 旧point AP | 新point AP | ΔAP |','|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["seed"]} | {row["old_point_ap"]:.8f} | {row["new_point_ap"]:.8f} | {row["delta_ap"]:+.8f} |')
    lines += ['',
        '三个seed的分数排序在该开发集上均改善，说明观察到的差异不限于保持排序的单调分数尺度变化。',
        'seed1019虽有较高AP改善，固定来源门控转移后的召回仍仅291/383=0.75979112，低于0.85；平均召回不能掩盖该失败。',
        'House17被反复使用，不作独立泛化声明；默认模型不切换，最终8/9/20保持隔离。',
        '',f'原交付manifest SHA256：{result["parent_manifest_sha256"]}']
    (output/'AP_DIAGNOSTIC.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    evidence = {path.name:digest(path) for path in output.iterdir() if path.is_file()}
    (output/'manifest.json').write_text(json.dumps(dict(artifacts=evidence,parent_manifest_sha256=result['parent_manifest_sha256']),indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run,args.output),ensure_ascii=False,indent=2))
