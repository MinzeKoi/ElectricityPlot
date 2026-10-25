# 微波炉中心TCN受控对照

离线模型使用未来上下文。默认不启用；功率沿用原A，无功率改善声明。

```json
{
  "status": "verified",
  "trained": true,
  "effect_improvement_established": false,
  "support": {
    "centers": 259806,
    "positive_centers": 849,
    "positive_centers_with_new_context": 849,
    "centers_with_new_context": 259804,
    "mean_new_observed_points": 90.97410760336558,
    "positive_mean_new_observed_points": 91.87043580683157,
    "stop_counts": {
      "left": {
        "missing_run": 5962,
        "radius": 253844
      },
      "right": {
        "boundary": 96,
        "missing_run": 5941,
        "radius": 253769
      }
    }
  },
  "benchmark": null,
  "actual_fits": 6,
  "development": {
    "selected_before_development": "TCN127",
    "improvement_candidate": true,
    "long_context_selected": true,
    "task4_source_preselected_comparison": {
      "selected_arm": "partial_mask",
      "freeze_sha256": "3f9f28e660b0f0f176f2fc8148e2024d3b9f0c46214548c9b80cdcccfaf11acd",
      "source_selection_sha256": "fcc7b51686e09c46fe223e047c569141481464d22331c035cded2caaa8adfd86"
    },
    "development_acceptance_passed": false,
    "declared_improvement": false,
    "means": {
      "A": {
        "precision": 0.04012808999869747,
        "recall": 0.9242819843342036,
        "f1": 0.07691297343567473,
        "fp": 8482.0
      },
      "TCN127": {
        "precision": 0.03560880469833081,
        "recall": 0.9895561357702349,
        "f1": 0.0687432362346322,
        "fp": 10267.0
      },
      "TCN31": {
        "precision": 0.03448139077949868,
        "recall": 0.9138381201044387,
        "f1": 0.06645358261392775,
        "fp": 9804.333333333334
      },
      "W50": {
        "precision": 0.04012808999869747,
        "recall": 0.9242819843342036,
        "f1": 0.07691297343567473,
        "fp": 8482.0
      },
      "partial_mask": {
        "precision": 0.034086879568519884,
        "recall": 0.9190600522193212,
        "f1": 0.06573388824743838,
        "fp": 9977.333333333334
      }
    },
    "selected_deltas": {
      "A": {
        "precision": -0.00451928530036666,
        "recall": 0.06527415143603132,
        "f1": -0.008169737201042526,
        "fp": 1785.0
      },
      "TCN31": {
        "precision": 0.0011274139188321292,
        "recall": 0.07571801566579628,
        "f1": 0.0022896536207044488,
        "fp": 462.66666666666606
      },
      "W50": {
        "precision": -0.00451928530036666,
        "recall": 0.06527415143603132,
        "f1": -0.008169737201042526,
        "fp": 1785.0
      },
      "partial_mask": {
        "precision": 0.0015219251298109263,
        "recall": 0.07049608355091375,
        "f1": 0.0030093479871938222,
        "fp": 289.66666666666606
      }
    },
    "common_support": {
      "42": {
        "n": 137523,
        "mask_sha256": "d7608bed577515bc4bd98ff1efe4c78a82798a96c9d1d3e5dd03bb90d53598a1",
        "timestamps_sha256": "342269261b9b59715ac588476545f50d6e875b7a01f6731745098c6c384a2ffe"
      },
      "1019": {
        "n": 137523,
        "mask_sha256": "d7608bed577515bc4bd98ff1efe4c78a82798a96c9d1d3e5dd03bb90d53598a1",
        "timestamps_sha256": "342269261b9b59715ac588476545f50d6e875b7a01f6731745098c6c384a2ffe"
      },
      "2026": {
        "n": 137523,
        "mask_sha256": "d7608bed577515bc4bd98ff1efe4c78a82798a96c9d1d3e5dd03bb90d53598a1",
        "timestamps_sha256": "342269261b9b59715ac588476545f50d6e875b7a01f6731745098c6c384a2ffe"
      }
    },
    "power_improvement_claim": false,
    "raw_power_and_other_devices_unchanged": true,
    "default_model_changed": false,
    "development_only_previously_inspected_house17": true,
    "final_houses_accessed": false,
    "additional_training_permitted": false
  }
}
```
