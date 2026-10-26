"""Frozen-result reporting only: confusion tables, FP spans, and exposure details."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode=True
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',required=True,type=Path)
parser.add_argument('--output-name',default='details_v2')
args=parser.parse_args()
run=args.run.resolve()
sys.path.insert(0,str(run/'source'))
from virtual_submeter.accuracy import microwave_source5_temporal_execution as e
from virtual_submeter.data.common import DataError,read_json,write_json,inventory,iso,sha256


def spans(t,flag,scores,aggregate,arm,seed):
    indices=np.flatnonzero(flag)
    if not len(indices): return []
    cut=np.r_[0,np.flatnonzero(np.diff(t[indices])!=8)+1,len(indices)]
    rows=[]
    for a,b in zip(cut[:-1],cut[1:]):
        ix=indices[a:b]
        rows.append(dict(arm=arm,seed=seed,start_utc=iso(int(t[ix[0]])),end_utc_exclusive=iso(int(t[ix[-1]]+8)),
            observed_false_positive_centers=len(ix),observed_grid_seconds=int(len(ix)*8),
            max_score=float(scores[ix].max()) if scores is not None else None,
            median_aggregate_w=float(np.median(aggregate[ix]))))
    return rows


if __name__=='__main__':
    e.evaluation_guard(run)
    complete=read_json(run/'evaluation_complete.json')
    e.check_artifacts(run/'evaluation',complete['artifacts'])
    if Path(args.output_name).name!=args.output_name: raise DataError('Output name must be a direct run child')
    dest=run/args.output_name
    dest.mkdir(exist_ok=False)
    truth=e.load_npz(run/'evaluation/truth_once.npz')
    pool=e.load_npz(run/'train.npz')
    confusions,segments,contributions,exposures=[],[],[],[]
    for arm in e.f.ARMS:
        for seed in e.f.SEEDS:
            model=run/f'models/{arm}/seed_{seed}'
            meta=read_json(model/'model.json')
            for key,value in meta['selected']['block_metrics'].items():
                confusions.append({**value,'arm':arm,'seed':seed,'block':key,'epoch':meta['selected']['epoch'],'threshold':meta['decision_threshold_probability']})
            data=e.load_npz(run/f'predictions/{arm}_{seed}.npz')
            mask=np.isfinite(truth['y'])&np.isfinite(data['scores'])
            segments+=spans(data['t'],mask&(truth['y']<200)&(data['state']==1),data['scores'],data['aggregate'],arm,seed)
            if arm=='source5':
                with (model/'batches.csv').open(encoding='utf-8',newline='') as stream: batches=list(csv.DictReader(stream))
                positive=sum(float(v['positive_loss_sum']) for v in batches)
                negative=sum(float(v['negative_loss_sum']) for v in batches)
                contributions.append(dict(seed=seed,updates=len(batches),positive_exposures=16*len(batches),negative_exposures=240*len(batches),
                    pos_weight=50,nominal_positive_weight_mass=50*16/(50*16+240),actual_positive_loss_sum=positive,
                    actual_negative_loss_sum=negative,actual_positive_loss_fraction=positive/(positive+negative)))
                exposure=e.load_npz(model/'exposure.npz')['count']
                grouped={}
                for i,count in enumerate(exposure):
                    key=int(pool['house'][i]),int(pool['t'][i]//86400),int(pool['event'][i])
                    row=grouped.setdefault(key,dict(seed=seed,house=['refit_2','refit_19','refit_5'][key[0]],utc_day=iso(key[1]*86400)[:10],
                        event=key[2],available_centers=0,unique_centers_seen=0,total_exposures=0))
                    row['available_centers']+=1;row['unique_centers_seen']+=int(count>0);row['total_exposures']+=int(count)
                exposures+=list(grouped.values())
    for seed in e.f.SEEDS:
        w=e.parent.reference_arrays(run,seed,'W50')
        mask=np.isfinite(truth['y'])&np.isfinite(w['raw'][:,1])
        segments+=spans(w['stamps'],mask&(truth['y']<200)&(w['raw'][:,1]>=200),None,w['aggregate'],'W50',seed)
    e.save_csv(dest/'source_selected_block_confusions.csv',confusions)
    e.save_csv(dest/'false_positive_spans.csv',segments)
    e.save_csv(dest/'loss_contributions.csv',contributions)
    e.save_csv(dest/'date_event_exposure.csv',exposures)
    longest=sorted(segments,key=lambda r:-r['observed_false_positive_centers'])[:40]
    lines=['# 冻结后补充明细','',
        '误报片段是连续可观测8秒网格，不等于独立物理事件；缺失或时断会断开。完整列表见false_positive_spans.csv。',
        '来源混淆使用末两块宏平均选模对应的共同阈值；CSV提供每块TP/FP/FN/TN。',
        'date_event_exposure.csv绑定真实家庭、UTC日、阈值段、可用/见过中心及总曝光。损失贡献是实际训练记录求和，不是理论类别权重。','',
        '|组|seed|开始UTC|结束UTC（不含）|误报点数|','|---|---:|---|---|---:|']
    for row in longest:
        lines.append(f'|{row["arm"]}|{row["seed"]}|{row["start_utc"]}|{row["end_utc_exclusive"]}|{row["observed_false_positive_centers"]}|')
    lines+=['','准备v1因路径参数类型错误终止于数据校验前，0新模型；保留未覆盖。正式训练有效目录为v2，3启动全部完成，无失败拟合或重试。',
        '代码准备目录microwave-source5-temporal-preparation-20261025-v1保持原样。新增的完整freeze/一次评价合成测试在训练进行中补充通过，初次启动前已通过真实fit_one训练/首次验证/保存加载与冻结门禁测试。',
        '本明细不改变任何模型、阈值、source预选、开发门槛或默认行为。']
    (dest/'DETAILS.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(dest/'manifest.json',dict(artifacts=inventory(dest),source_selection_sha256=sha256(run/'source_selection.json'),
        evaluation_complete_sha256=sha256(run/'evaluation_complete.json')))
    print(json.dumps(dict(fp_spans=len(segments),source_confusions=len(confusions),exposure_rows=len(exposures),output=str(dest)),ensure_ascii=False))
