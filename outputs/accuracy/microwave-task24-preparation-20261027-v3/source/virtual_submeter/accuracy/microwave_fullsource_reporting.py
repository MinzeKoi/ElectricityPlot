"""Report semantics fixed by coordinator before any full-source development scores."""


def improvement_flags(summary):
    passed = bool(summary["development_acceptance_passed"] and summary["selected_before_development"] is not None)
    return {"declared_improvement": passed,
            "sampling_or_background_improvement_supported": passed and summary["selected_before_development"] != "natural",
            "improvement_candidate": summary["improvement_candidate"]}
