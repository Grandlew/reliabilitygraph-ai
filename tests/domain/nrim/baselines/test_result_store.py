from app.domain.nrim.baselines.models import (
    BaselineEvaluationResult,
    BaselineName,
    RankingMetricSummary,
)
from app.domain.nrim.baselines.result_store import (
    build_result_markdown,
    save_evaluation_result,
)


def summary() -> RankingMetricSummary:
    return RankingMetricSummary(
        window_count=10,
        evaluated_faulty_windows=5,
        mrr=0.6,
        hits_at_1=0.4,
        hits_at_3=0.8,
        mean_rank=2.0,
        median_rank=2.0,
        healthy_window_count=5,
        healthy_abstention_rate=0.8,
        healthy_false_selection_rate=0.2,
        faulty_coverage=0.9,
        selective_mrr=0.67,
        mean_runtime_ms=1.5,
    )


def make_result() -> BaselineEvaluationResult:
    return BaselineEvaluationResult(
        baseline=(
            BaselineName.HYBRID_ENGINEERING
        ),
        benchmark_fingerprint="a" * 64,
        abstention_threshold=0.4,
        validation=summary(),
        test=summary(),
        ood_test=summary(),
        configuration={
            "synthetic": True
        },
    )


def test_markdown_contains_fingerprint() -> None:
    markdown = build_result_markdown(
        make_result()
    )

    assert "a" * 64 in markdown
    assert "hybrid_engineering" in markdown
    assert "synthetic" in markdown.lower()


def test_result_files_are_saved(
    tmp_path,
) -> None:
    json_path, markdown_path = (
        save_evaluation_result(
            result=make_result(),
            output_dir=tmp_path,
        )
    )

    assert json_path.exists()
    assert markdown_path.exists()
