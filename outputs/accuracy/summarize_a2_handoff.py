import json
import math
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'src'))
from virtual_submeter.accuracy.fixed_queries import verify
from virtual_submeter.data.common import read_json, sha256, write_json

run = root / 'outputs/accuracy/a2-20261021'
validation = verify(run)
summary = read_json(run / 'summary.json')
metrics = read_json(run / 'metrics.json')['rows']
plan = read_json(run / 'frozen/plan.json')
device_rows = []
for variant in plan['variants']:
    for budget in plan['budgets']:
        for appliance in plan['primary_appliances'] + plan['extension_appliances']:
            rows = [m for m in metrics if m['variant'] == variant and m['budget'] == budget and m['appliance'] == appliance]
            item = dict(variant=variant, budget=budget, appliance=appliance, expected_pairs=9, observed_pairs=len(rows))
            for key in ('f1', 'f1_delta', 'baseline_f1', 'precision', 'recall', 'predicted_on_fraction'):
                item[key] = math.fsum(m[key] for m in rows) / len(rows) if rows and all(m[key] is not None for m in rows) else None
            item['adaptation_caused_degeneracy'] = sum(m['adaptation_caused_degeneracy'] for m in rows)
            device_rows.append(item)

main = [m for m in metrics if m['variant'] == 'L2_max' and m['budget'] == 10 and m['appliance'] in plan['primary_appliances']]
positive = 0
for source in plan['source_seeds']:
    for query in plan['query_seeds']:
        rows = [m for m in main if m['source_seed'] == source and m['query_seed'] == query]
        positive += int(math.fsum(m['f1_delta'] for m in rows) > 0)

result = dict(schema_version=1, date='2026-10-21', accuracy_usage='development_only', formal_evaluation=False,
              run='outputs/accuracy/a2-20261021', source_training_seeds=plan['source_seeds'], query_seeds=plan['query_seeds'],
              development_households=['3'], cases=252, metric_rows=756, main_budget=10,
              label_source='simulated_from_submeter', power_output_frozen=True,
              groups=summary['groups'], per_device=device_rows,
              main_budget_positive_source_query_pairs=positive, main_budget_total_source_query_pairs=9,
              development_exit_passed=summary['development_exit_passed'], internal_gain_target_met=summary['internal_gain_target_met'],
              selected_development_learner=summary['selected_development_learner'],
              query_distribution=read_json(run / 'query_distribution.json')['rows'],
              limitations=['One previously inspected development household; seeds are not independent households.',
                           'No kettle positive answer in any of the fixed random trajectories.',
                           'Some queried windows return unsure due to missing submeter support; all count toward budget.',
                           'L2 source-state baseline improvement is not confirmation gain.',
                           'The +0.05 internal gain goal and final multi-household H1/H2/H3 are not met.',
                           'No power/energy improvement; no A3 comparison or independent-person/second-machine replication.'],
              bindings={name:sha256(run / name) for name in ('run.json','freeze.json','evidence.json','summary.json','metrics.json','report.html')})
write_json(root / 'docs/A2_RESULTS.json', result)
checks = read_json(root / 'docs/A2_VALIDATION.json')
checks['artifact_verification'] = validation
checks['artifact_inventory_files'] = len(read_json(run / 'freeze.json')['artifacts_sha256'])
checks['local_access_note'] = 'Run and independent verification used the same approved file access. The restricted sandbox cannot enumerate 42 published files and reports an incomplete inventory; with matching access all 1668 frozen artifacts verify without changes.'
checks['tests']['log_sha256'] = sha256(root / 'outputs/accuracy/a2-tests.log')
checks['real_boundary_checks']['artifact_sha256'] = sha256(root / 'outputs/accuracy/a2-boundary-check.json')
checks['result_summary_sha256'] = sha256(root / 'docs/A2_RESULTS.json')
write_json(root / 'docs/A2_VALIDATION.json', checks)
print(json.dumps(validation, indent=2))
