"""Adapt Task24's synthetic boundary tests to the Task25 fixed quota."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for name in ('budget_draft', 'source_draft', 'train_boundary'):
    text = (root / f'tests/test_microwave_task24_{name}.py').read_text(encoding='utf-8')
    text = text.replace('run_microwave_task24_draft.py', 'run_microwave_task25.py')
    text = text.replace('microwave_task24_multihouse_v1', 'microwave_task25_positive_share_v1').replace('exact Task24 budget', 'exact Task25 budget')
    if name == 'source_draft':
        before = "            aggregate = run / 'aggregate.csv'\n"
        after = "            task24_source = run / 'task24_source.json'\n            task24_source.write_text(json.dumps({'rows': []}), encoding='utf-8')\n" + before
        assert text.count(before) == 1
        text = text.replace(before, after)
        text = text.replace("[control, aggregate, *map(Path, validation.values()), Path(gate.__file__)]", "[control, aggregate, task24_source, *map(Path, validation.values()), Path(gate.__file__)]")
        text = text.replace("'source5_aggregate': str(aggregate), 'validation': validation", "'source5_aggregate': str(aggregate), 'task24_selection': str(task24_source), 'validation': validation")
    if name == 'train_boundary':
        text = text.replace('            event = np.zeros(count, np.int64)\n', '            house[old_n:] = 3\n            event = np.zeros(count, np.int64)\n')
        before = "            positive = np.flatnonzero(all_y >= 200)[:16]\n            negative = np.flatnonzero(all_y < 200)[:240]\n            batch = np.broadcast_to(np.r_[positive, negative], (3528, 256))\n"
        after = "            old_positive = np.flatnonzero((all_y >= 200) & (house <= 2))[:11]\n            new_positive = np.flatnonzero((all_y >= 200) & (house >= 3))[:6]\n            negative = np.flatnonzero(all_y < 200)[:240]\n            cycle = np.stack([np.r_[old_positive[:n], new_positive[:16-n], negative] for n in (11,11,10)])\n            batch = np.tile(cycle, (1176, 1))\n"
        assert text.count(before) == 1
        text = text.replace(before,after)
    (root / f'tests/test_microwave_task25_{name}.py').write_text(text,encoding='utf-8')
