"""Read-only verification of immutable source-only audit artifacts."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from virtual_submeter.data.common import sha256,utc

def verify(run):
    receipt=json.loads((run/'complete.json').read_text())
    for rel,digest in receipt['artifacts'].items():
        assert sha256(run/rel)==digest,rel
    audit=json.loads((run/'audit.json').read_text())
    daily=json.loads((run/'daily_support.json').read_text())['rows']
    events=json.loads((run/'events.json').read_text())['events']
    results=[]
    for h,info in audit['households'].items():
        assert sha256(ROOT/info['original_path'])==info['original_sha256']
        for channel,meta in info['replay'].items():
            assert sha256(ROOT/meta['old_path'])==meta['sha256']
            assert sha256(ROOT/meta['input_path'])==meta['sha256']
        data=np.load(run/f'{h}_support.npz')
        t=data['t'];yrows=list(csv.DictReader((ROOT/info['replay']['microwave']['input_path']).open(encoding='utf-8')))
        arows=list(csv.DictReader((ROOT/info['replay']['aggregate']['input_path']).open(encoding='utf-8')))
        y=np.array([float(r['power_w']) if r['power_w'] else np.nan for r in yrows])
        a=np.array([float(r['aggregate_power_w']) if r['aggregate_power_w'] else np.nan for r in arows])
        # Independently derive C31 validity using prefix missing counts, not sliding_window_view.
        bad=np.r_[0,np.cumsum(~np.isfinite(a))];valid=np.zeros(len(a),bool)
        centers=np.arange(15,len(a)-15);valid[centers]=(bad[centers+16]-bad[centers-15])==0
        np.testing.assert_array_equal(valid,data['complete_context'])
        np.testing.assert_array_equal(valid&np.isfinite(y),data['usable'])
        rows=[r for r in daily if r['household']==h]
        assert sum(r['usable_windows'] for r in rows)==info['total_usable_windows']
        assert sum(r['usable_on_centers'] for r in rows)==info['total_positive_windows']
        for r in rows:
            assert sum(r['window_loss_exclusive'].values())==r['grid_bins']
            assert sum(r['on_window_loss_exclusive'].values())==r['observed_on_centers']
            assert r['label_valid_bins_before_issues']-r['label_bins_lost_to_issues_after_resampling']==r['label_valid_bins']
        es=[e for e in events if e['household']==h]
        for family in ['historical','recommended']:
            previous_context_end=None;seen=set()
            for role in ['train','validation','outer']:
                name=family+'_'+role;b=info['candidate_blocks'][name];mask=data['mask_'+name];idx=np.flatnonzero(mask)
                assert len(idx)==b['windows'];assert int((mask&(y>=200)).sum())==b['positive_windows']
                assert np.all(t[idx]-120>=utc(b['start'])) and np.all(t[idx]+128<=utc(b['end']))
                if previous_context_end is not None:assert previous_context_end<=t[idx[0]]-120
                previous_context_end=int(t[idx[-1]])+128
                supported={e['id'] for e in es if mask[e['a']:e['b']].any()}
                assert not seen&supported;seen|=supported
                complete=[e['id'] for e in es if mask[e['a']:e['b']].all() and e['observed_off_boundaries'] and
                          t[e['a']]-8>=utc(b['start']) and t[e['b']-1]+16<=utc(b['end'])]
                assert complete==b['complete_event_ids']
        results.append({'household':h,'input_hashes_checked':True,'independent_prefix_window_check':True,
            'daily_loss_accounting':True,'issues_stage_accounting':True,'both_split_context_event_checks':True})
    for rel,digest in audit['old_run_hashes'].items():
        assert sha256(ROOT/'outputs/accuracy/microwave-diagnosis-20261025-v2'/rel)==digest
    return {'status':'passed','audited_artifacts':len(receipt['artifacts']),'checks':results,
            'old_diagnostic_receipts_unchanged':True,'no_model_fits':True}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(verify(args.run.resolve()),ensure_ascii=False,indent=2))
