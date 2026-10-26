"""Record the new explicit human Task18 budget; never trains itself."""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'outputs/accuracy/microwave-expanded-guard95-preparation-20261026-v3'
sys.path.insert(0,str(run/'source'))
from virtual_submeter.accuracy import microwave_expanded_guard95 as e
expected='11dd4f8d9fdf68d2740c4b643a869ae462e21f4518eb309aacdb89eb92d11040'
e.require(e.sha256(run/'registration.json')==expected,'Wrong Task18 registration')
e.require(not any((run/p).exists() for p in ('training_started.json','models','human_authorization.json','execution_identity.json')),'Existing execution: inspect session/state, never restart')
receipt=dict(authorized=True,approval_origin='human',experiment=e.EXPERIMENT,
    additional_new_fits=3,maximum_updates=19404,seeds=e.SEEDS,registration_sha256=expected,
    approved_rule='expanded_guard95_fixed_L95_H95_radius15',
    human_authorization_quote='允许新增3模型／19,404次更新',
    human_authorization_reference='01a16d7b-478e-7780-a999-88abe71e8c94/request_user_input_async/call_56eaec9ecea54fccbd0e0c83a2a084b2',
    authorization_scope='Frozen Task18 v3 train→source→source-qualified only freeze (12 ordinary CLI)→one parent-cache evaluation→readonly verification/report; no control retraining, no retry/additional fits, no default/UI changes, final houses 8/9/20 isolated')
e.fresh(run/'human_authorization.json',receipt)
e.authorize(run,run/'human_authorization.json')
verified=e.verify_preparation(run)
e.fresh(run/'execution_identity.json',dict(experiment=e.EXPERIMENT,registration_sha256=expected,
    preparation_manifest_sha256=e.sha256(run/'final_delivery_manifest.json'),
    authorization_sha256=e.sha256(run/'human_authorization.json'),
    frozen_implementation_sha256=e.sha256(Path(e.__file__)),
    authorized_new_fits=3,authorized_updates=19404,control_retraining=False,
    verification_before_training=verified))
print(json.dumps(dict(status='new_human_budget_bound_not_yet_started',registration_sha256=expected,authorization_sha256=e.sha256(run/'human_authorization.json')),ensure_ascii=False,indent=2))
