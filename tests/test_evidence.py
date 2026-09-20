from guardian_of_truth.api_client import AuditPayload
from guardian_of_truth.claims import extract_claims
import csv
import json

import pandas as pd

from guardian_of_truth.drift_overlay import answer_shape, apply_drift_overlay_frame
from guardian_of_truth.evidence import (
    EvidenceRetriever,
    EvidenceSnippet,
    RETRIEVER_VERSION,
    build_kb,
    controlled_recall_query,
    is_generic_subject_token,
    query_terms_for_retrieval,
    query_terms_for_retrieval_v4,
    subject_terms_from_prompt,
    verify_against_evidence,
)
from guardian_of_truth.entity_overlay import apply_entity_overlay_frame, extract_core_entities
from guardian_of_truth.feature_extractor import FeatureExtractor
from scripts.analyze_overlay_effects import analyze_overlay_effects
from scripts.apply_refute_overlay_v3 import apply_overlay
from scripts.apply_drift_overlay_v6 import apply_overlay as apply_drift_overlay_v6
from guardian_of_truth.refute_overlay import apply_overlay_frame, expected_answer_kind, overlay_decision
from scripts.apply_refute_overlay_v4 import apply_overlay as apply_overlay_v4
from scripts.export_unlabeled_queries import export_queries
from scripts.fetch_wikipedia_kb import build_queries
from scripts.report_evidence_coverage import build_coverage_report
from scripts.export_targeted_kb_queries_v8 import export_queries as export_targeted_queries_v8


def test_claim_extraction_handles_markdown_tail_numbers_and_entities() -> None:
    claims = extract_claims("**Санкт-Петербург** основан в 1703 году. Основателем был Пётр Первый.")

    assert claims.core_answer.startswith("Санкт-Петербург")
    assert "1703" in claims.years
    assert len(claims.claims) == 2
    assert claims.claims[0].is_core


def test_build_kb_rejects_label_leakage_column(tmp_path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("prompt,answer,is_hallucination\nq,a,0\n", encoding="utf-8")

    try:
        build_kb([source], tmp_path / "kb.sqlite")
    except ValueError as exc:
        assert "forbidden columns" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected leakage guard to reject labels")


def test_export_unlabeled_queries_strips_forbidden_columns(tmp_path) -> None:
    source = tmp_path / "public.csv"
    source.write_text(
        "prompt,model_answer,is_hallucination,correct_answer,comment\nWho wrote Hamlet?,Shakespeare,0,William Shakespeare,note\n",
        encoding="utf-8",
    )
    output = tmp_path / "queries.csv"

    summary = export_queries(source, output)
    exported = pd.read_csv(output)

    assert summary["rows"] == 1
    assert list(exported.columns) == ["row_key", "prompt", "model_answer"]
    assert "correct_answer" not in exported.columns
    assert "is_hallucination" not in exported.columns


def test_fetch_query_builder_returns_multiple_query_types() -> None:
    queries = build_queries(
        {
            "prompt": "В каком году был основан Санкт-Петербург Петром Первым?",
            "model_answer": "Санкт-Петербург был основан в 1703 году.",
        }
    )

    query_types = {item["query_type"] for item in queries}
    assert len(queries) >= 3
    assert "answer_core_prompt_keywords" in query_types
    assert "prompt_keywords_answer_number" in query_types


def test_evidence_retriever_returns_supported_and_refuted_features(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"prompt":"В каком году был основан Санкт-Петербург?","answer":"Санкт-Петербург был основан в 1703 году.","source":"seed"}\n',
        encoding="utf-8",
    )
    db = tmp_path / "kb.sqlite"
    summary = build_kb([source], db, min_chars=80)
    assert summary["snippets"] >= 1

    retriever = EvidenceRetriever(db, retriever_version="fts5_rules_v3")
    supported = retriever.audit("В каком году был основан Санкт-Петербург?", "Санкт-Петербург был основан в 1703 году.")
    refuted = retriever.audit("В каком году был основан Санкт-Петербург?", "Санкт-Петербург был основан в 1492 году.")

    assert supported.retrieval_hit_count >= 1
    assert supported.core_supported == 1.0
    assert refuted.core_refuted == 1.0


def test_build_kb_indexes_short_wikipedia_extract(tmp_path) -> None:
    source = tmp_path / "wiki.jsonl"
    text = "Санкт-Петербург — город федерального значения в России. Он был основан Петром I в 1703 году."
    assert 120 > len(text) >= 80
    source.write_text(
        '{"title":"Санкт-Петербург","text":"' + text + '","source":"wikipedia:ru","lang":"ru"}\n',
        encoding="utf-8",
    )
    db = tmp_path / "kb.sqlite"
    summary = build_kb([source], db, min_chars=80)

    assert summary["snippets"] == 1


def test_cache_key_changes_with_retriever_version_or_kb_version(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"prompt":"q","answer":"a","text":"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"}\n', encoding="utf-8")
    db1 = tmp_path / "kb1.sqlite"
    db2 = tmp_path / "kb2.sqlite"
    build_kb([source], db1, kb_version="local_wiki_v3")
    build_kb([source], db2, kb_version="local_wiki_v3_retry1")

    r1 = EvidenceRetriever(db1)
    r2 = EvidenceRetriever(db2)
    assert r1.cache_key("q", "a") != r2.cache_key("q", "a")
    before = r1.cache_key("q", "a")
    monkeypatch.setattr("guardian_of_truth.evidence.RETRIEVER_VERSION", RETRIEVER_VERSION + "_test")
    assert before != EvidenceRetriever(db1).cache_key("q", "a")


def test_v7_subject_extraction_removes_generic_question_words() -> None:
    terms = subject_terms_from_prompt("Какое название получила техника в эксперименте AlphaGo Zero?")

    assert "какое" not in terms
    assert "название" not in terms
    assert "техника" not in terms
    assert "эксперимент" not in terms
    assert "alphago" in terms
    assert "zero" in terms
    assert is_generic_subject_token("год")
    assert is_generic_subject_token("1909")


def test_v7_query_construction_avoids_full_prompt_or_query() -> None:
    prompt = "Какую актрису изображал плакат в фильме «Побег из Шоушенка»?"
    answer = "Плакат изображал Риту Хейворт."
    queries = query_terms_for_retrieval(prompt, answer)

    assert 1 <= len(queries) <= 5
    assert all(len(query.split()) <= 12 for query in queries)
    assert all("какую" not in query.lower() for query in queries)
    assert any("побег" in query.lower() and "шоушенка" in query.lower() for query in queries)


def test_v8_targeted_query_export_strips_forbidden_columns_and_filters(tmp_path) -> None:
    source = tmp_path / "kb_recall.csv"
    pd.DataFrame(
        [
            {
                "row_key": "r1",
                "prompt": "Какую актрису изображал плакат в фильме «Побег из Шоушенка»?",
                "model_answer": "Плакат изображал Риту Хейворт.",
                "is_hallucination": 1,
                "correct_answer": "forbidden",
                "comment": "forbidden",
            },
            {
                "row_key": "r2",
                "prompt": "В каком году 1900?",
                "model_answer": "1900",
                "is_hallucination": 0,
            },
        ]
    ).to_csv(source, index=False)
    output = tmp_path / "queries.jsonl"

    summary = export_targeted_queries_v8(source, output, max_queries_per_row=5)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["queries"] >= 1
    assert all("correct_answer" not in record and "is_hallucination" not in record for record in records)
    assert all(len(record["query"].split()) <= 12 for record in records)
    assert all(not record["query"].isdigit() for record in records)
    assert any(record["query_type"] == "quoted_prompt_title" for record in records)


def test_v8_query_helpers_keep_precision_and_controlled_recall() -> None:
    prompt = "Какую актрису изображал плакат в фильме «Побег из Шоушенка»?"
    answer = "Плакат изображал Риту Хейворт."
    queries = query_terms_for_retrieval_v4(prompt, answer)

    assert 1 <= len(queries) <= 5
    assert all(len(query.split()) <= 12 for query in queries)
    assert controlled_recall_query(prompt)
    assert any("побег" in query and "шоушенка" in query for query in queries)


def test_v7_retriever_rejects_numeric_only_overlap(tmp_path) -> None:
    source = tmp_path / "kb.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"title":"Москва","text":"Москва была основана в 1147 году. История города подробно описана в летописях.","source":"wiki"}',
                '{"title":"Санкт-Петербург","text":"Санкт-Петербург был основан Петром I в 1703 году. Город находится в России.","source":"wiki"}',
            ]
        ),
        encoding="utf-8",
    )
    db = tmp_path / "kb.sqlite"
    build_kb([source], db, min_chars=40)
    retriever = EvidenceRetriever(db, retriever_version="fts5_rules_v3")

    snippets = retriever.retrieve("В каком году был основан Санкт-Петербург?", "Санкт-Петербург был основан в 1147 году.")

    assert snippets
    assert snippets[0].title == "Санкт-Петербург"
    assert all(snippet.title != "Москва" for snippet in snippets)


def test_v7_aligned_evidence_reports_reason_and_rejects_weak_overlap() -> None:
    unrelated = EvidenceSnippet(
        id=1,
        title="Москва",
        text="Москва была основана в 1147 году и находится в России.",
        source="wiki",
        score=1.0,
    )
    aligned = EvidenceSnippet(
        id=2,
        title="Санкт-Петербург",
        text="Санкт-Петербург был основан Петром I в 1703 году.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1492 году.",
        [unrelated, aligned],
        alignment_version="fts5_rules_v3",
    )
    compact = json.loads(audit.compact_json)

    assert audit.aligned_hit_count == 1
    assert compact["aligned_snippets"][0]["alignment_reason"] == "title_subject_overlap"
    assert compact["rejected_top_snippets"][0]["reason"] == "weak_subject_alignment"


def test_v8_compact_json_separates_raw_aligned_and_rejected() -> None:
    unrelated = EvidenceSnippet(
        id=1,
        title="Москва",
        text="Москва была основана в 1147 году и находится в России.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1492 году.",
        [unrelated],
        alignment_version="fts5_rules_v4",
        retriever_version="fts5_rules_v4",
        kb_version="local_wiki_v4",
        queries=["санкт петербург основан"],
    )
    compact = json.loads(audit.compact_json)

    assert audit.status == "ok"
    assert audit.retrieval_hit_count == 1
    assert audit.aligned_hit_count == 0
    assert compact["retriever_version"] == "fts5_rules_v4"
    assert compact["kb_version"] == "local_wiki_v4"
    assert compact["queries"] == ["санкт петербург основан"]
    assert compact["raw_snippets"]
    assert compact["aligned_snippets"] == []
    assert compact["rejected_snippets"][0]["reason"] == "weak_subject_alignment"


def test_v9_strong_alignment_accepts_quoted_prompt_title() -> None:
    snippet = EvidenceSnippet(
        id=1,
        title="Hamlet",
        text="Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601.",
        source="wikipedia:en",
        score=1.0,
    )
    audit = verify_against_evidence(
        'Who wrote "Hamlet"?',
        "Christopher Marlowe.",
        [snippet],
        alignment_version="fts5_rules_v5",
        retriever_version="fts5_rules_v5",
        kb_version="local_wiki_v4",
    )
    compact = json.loads(audit.compact_json)

    assert audit.retrieval_hit_count == 1
    assert audit.aligned_hit_count == 1
    assert compact["overlay_eligible_snippets"][0]["alignment_tier"] == "strong"
    assert compact["overlay_eligible_snippets"][0]["alignment_reason"] == "quoted_prompt_title_in_title"


def test_v9_medium_alignment_affects_features_not_overlay() -> None:
    snippet = EvidenceSnippet(
        id=1,
        title="обломов",
        text="Роман Обломов рассказывает историю героя и его окружения.",
        source="wikipedia:ru",
        score=1.0,
    )
    audit = verify_against_evidence(
        "какой автор написал роман обломов история",
        "Иван Гончаров написал роман.",
        [snippet],
        alignment_version="fts5_rules_v5",
    )
    compact = json.loads(audit.compact_json)

    assert audit.status == "ok"
    assert audit.retrieval_hit_count == 1
    assert audit.aligned_hit_count == 0
    assert compact["feature_snippets"][0]["alignment_tier"] == "medium"
    assert compact["overlay_eligible_snippets"] == []
    assert audit.top_evidence_overlap > 0


def test_v9_weak_alignment_diagnostics_only_and_no_feature_overlap() -> None:
    snippet = EvidenceSnippet(
        id=1,
        title="Москва",
        text="В тексте случайно упоминаются основан петербург, но статья о другом городе.",
        source="wikipedia:ru",
        score=1.0,
    )
    audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1492 году.",
        [snippet],
        alignment_version="fts5_rules_v5",
    )
    compact = json.loads(audit.compact_json)

    assert audit.status == "ok"
    assert audit.retrieval_hit_count == 1
    assert audit.aligned_hit_count == 0
    assert audit.top_evidence_overlap == 0
    assert compact["weak_aligned_snippets"][0]["alignment_tier"] == "weak"
    assert compact["feature_snippets"] == []


def test_v9_numeric_only_and_answer_tail_only_are_rejected() -> None:
    numeric = EvidenceSnippet(id=1, title="1492", text="1492 1492 1492", source="wiki", score=1.0)
    answer_tail = EvidenceSnippet(id=2, title="Christopher Marlowe", text="Christopher Marlowe was an English playwright.", source="wiki", score=1.0)

    numeric_audit = verify_against_evidence("When was Hamlet written?", "1492", [numeric], alignment_version="fts5_rules_v5")
    tail_audit = verify_against_evidence("Who wrote Hamlet?", "Christopher Marlowe.", [answer_tail], alignment_version="fts5_rules_v5")

    numeric_compact = json.loads(numeric_audit.compact_json)
    tail_compact = json.loads(tail_audit.compact_json)
    assert numeric_audit.aligned_hit_count == 0
    assert numeric_compact["feature_snippets"] == []
    assert numeric_compact["rejected_snippets"][0]["reason"] == "numeric_only_overlap"
    assert tail_audit.aligned_hit_count == 0
    assert tail_compact["feature_snippets"] == []
    assert tail_compact["rejected_snippets"][0]["reason"] == "answer_tail_only"


def test_v9_year_refute_uses_only_strong_overlay_evidence() -> None:
    medium = EvidenceSnippet(
        id=1,
        title="петербург",
        text="Петербург город России без даты основания в этом фрагменте.",
        source="wiki",
        score=1.0,
    )
    strong = EvidenceSnippet(
        id=2,
        title="Санкт-Петербург",
        text="Санкт-Петербург был основан Петром I в 1703 году.",
        source="wiki",
        score=1.0,
    )

    medium_audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1492 году.",
        [medium],
        alignment_version="fts5_rules_v5",
    )
    strong_audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1492 году.",
        [strong],
        alignment_version="fts5_rules_v5",
    )

    assert medium_audit.retrieval_hit_count == 1
    assert medium_audit.aligned_year_refuted_count == 0
    assert strong_audit.aligned_hit_count == 1
    assert strong_audit.aligned_year_refuted_count == 1


def test_v10_year_relation_accepts_matching_predicate_conflict() -> None:
    snippet = EvidenceSnippet(
        id=1,
        title="Санкт-Петербург",
        text="Санкт-Петербург был основан Петром I в 1703 году.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1492 году.",
        [snippet],
        alignment_version="fts5_rules_v6",
        retriever_version="fts5_rules_v6",
    )
    compact = json.loads(audit.compact_json)

    assert audit.aligned_year_refuted_count == 1
    assert compact["relation_validation_version"] == "relation_refute_v10"
    assert compact["prompt_relation_kind"] == "founded"
    assert compact["relation_conflicts"][0]["value"] == "1703"


def test_v10_year_relation_rejects_unrelated_year_without_predicate() -> None:
    snippet = EvidenceSnippet(
        id=1,
        title="Санкт-Петербург",
        text="Санкт-Петербург находится в России. В 1914 году город был переименован.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1492 году.",
        [snippet],
        alignment_version="fts5_rules_v6",
    )
    compact = json.loads(audit.compact_json)

    assert audit.aligned_year_refuted_count == 0
    assert compact["relation_conflicts"] == []
    assert compact["rejected_relation_candidates"][0]["decision"] == "reject_no_predicate"


def test_v10_year_relation_rejects_broad_year_list() -> None:
    snippet = EvidenceSnippet(
        id=1,
        title="Hamlet",
        text="Hamlet editions are listed for 1599, 1601, 1623, 1709 and 1772 in this overview.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        'When was "Hamlet" written?',
        "1492",
        [snippet],
        alignment_version="fts5_rules_v6",
    )
    compact = json.loads(audit.compact_json)

    assert audit.aligned_year_refuted_count == 0
    assert any(item["decision"] == "reject_broad_list" for item in compact["rejected_relation_candidates"])


def test_v10_year_relation_rejects_when_answer_year_supported() -> None:
    snippet = EvidenceSnippet(
        id=1,
        title="Санкт-Петербург",
        text="Санкт-Петербург был основан Петром I в 1703 году. Некоторые источники также обсуждают 1712 год.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1703 году.",
        [snippet],
        alignment_version="fts5_rules_v6",
    )
    compact = json.loads(audit.compact_json)

    assert audit.aligned_year_refuted_count == 0
    assert compact["relation_conflicts"] == []


def test_v10_count_relation_accepts_compatible_unit_conflict() -> None:
    snippet = EvidenceSnippet(
        id=1,
        title="Mars",
        text="Mars has 2 moons, Phobos and Deimos.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        "How many moons does Mars have?",
        "Mars has 3 moons.",
        [snippet],
        alignment_version="fts5_rules_v6",
    )
    compact = json.loads(audit.compact_json)

    assert audit.aligned_number_refuted_count == 1
    assert compact["relation_conflicts"][0]["kind"] == "count"
    assert compact["relation_conflicts"][0]["unit_terms"] == ["moons"]


def test_v10_count_relation_rejects_wrong_unit_and_years() -> None:
    snippet = EvidenceSnippet(
        id=1,
        title="Mars",
        text="Mars was observed in 1877 and has several mission pages with version 2 listed.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        "How many moons does Mars have?",
        "Mars has 3 moons.",
        [snippet],
        alignment_version="fts5_rules_v6",
    )
    compact = json.loads(audit.compact_json)

    assert audit.aligned_number_refuted_count == 0
    assert compact["relation_conflicts"] == []


def test_v10_retriever_cache_key_differs_from_v5(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"title":"Mars","text":"Mars has two moons, Phobos and Deimos.","source":"wiki"}\n', encoding="utf-8")
    db = tmp_path / "kb.sqlite"
    build_kb([source], db, min_chars=20)

    v5 = EvidenceRetriever(db, retriever_version="fts5_rules_v5")
    v6 = EvidenceRetriever(db, retriever_version="fts5_rules_v6")

    assert v5.cache_key("How many moons does Mars have?", "3") != v6.cache_key("How many moons does Mars have?", "3")


def test_coverage_report_counts_missing_rows(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"prompt":"Who wrote Hamlet?","answer":"William Shakespeare wrote Hamlet.","text":"William Shakespeare wrote the tragedy Hamlet. Hamlet is a play by William Shakespeare written around 1600.","source":"wiki"}\n',
        encoding="utf-8",
    )
    db = tmp_path / "kb.sqlite"
    build_kb([source], db, min_chars=80)
    csv_path = tmp_path / "queries.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_key", "prompt", "model_answer"])
        writer.writeheader()
        writer.writerow({"row_key": "1", "prompt": "Who wrote Hamlet?", "model_answer": "William Shakespeare wrote Hamlet."})
        writer.writerow({"row_key": "2", "prompt": "When was Imaginary City founded?", "model_answer": "Imaginary City was founded in 1777."})

    report, missing = build_coverage_report(csv_path, db)

    assert report["rows"] == 2
    assert report["rows_with_evidence"] >= 1
    assert len(missing) >= 1


def test_refute_overlay_never_decreases_and_requires_high_confidence(tmp_path) -> None:
    scored = tmp_path / "scored.csv"
    pd.DataFrame(
        [
            {
                "prompt": "В каком году был основан Санкт-Петербург?",
                "model_answer": "В 1492 году.",
                "is_hallucination": 1,
                "is_hallucination_proba": 0.2,
                "evidence_retrieval_hit_count": 2,
                "evidence_top_evidence_overlap": 0.2,
                "evidence_core_refuted": 1,
                "evidence_claim_refuted_count": 1,
            },
            {
                "prompt": "Who wrote Hamlet?",
                "model_answer": "William Shakespeare.",
                "is_hallucination": 0,
                "is_hallucination_proba": 0.7,
                "evidence_retrieval_hit_count": 1,
                "evidence_top_evidence_overlap": 0.2,
                "evidence_core_refuted": 1,
                "evidence_claim_refuted_count": 1,
            },
        ]
    ).to_csv(scored, index=False)
    out = tmp_path / "overlay.csv"

    report = apply_overlay(scored, out)
    frame = pd.read_csv(out)

    assert report["never_decreased"]
    assert frame.loc[0, "is_hallucination_proba"] == 0.82
    assert frame.loc[1, "is_hallucination_proba"] == 0.7


def test_v4_expected_answer_detector_skips_contextual_year_prompts() -> None:
    assert expected_answer_kind("В каком журнале в 1853 году был опубликован рассказ?") == "none"
    assert expected_answer_kind("В каком фильме снимался актёр?") == "none"
    assert expected_answer_kind("Какая техника применялась в 1910 году?") == "none"
    assert expected_answer_kind("Какое название получила экспедиция?") == "none"
    assert expected_answer_kind("В каком году был основан Санкт-Петербург?") == "year"
    assert expected_answer_kind("How many moons does Mars have?") == "count"


def test_aligned_year_conflict_ignores_unrelated_snippet() -> None:
    unrelated = EvidenceSnippet(
        id=1,
        title="Москва",
        text="Москва была основана в 1147 году и находится в России.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1492 году.",
        [unrelated],
    )

    assert audit.aligned_hit_count == 0
    assert audit.aligned_year_refuted_count == 0


def test_aligned_year_conflict_triggers_for_subject_snippet() -> None:
    aligned = EvidenceSnippet(
        id=1,
        title="Санкт-Петербург",
        text="Санкт-Петербург был основан Петром I в 1703 году.",
        source="wiki",
        score=1.0,
    )
    audit = verify_against_evidence(
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1492 году.",
        [aligned],
    )

    assert audit.aligned_hit_count == 1
    assert audit.aligned_year_refuted_count == 1
    row = pd.Series(
        {
            "prompt": "В каком году был основан Санкт-Петербург?",
            "is_hallucination_proba": 0.2,
            "evidence_aligned_hit_count": audit.aligned_hit_count,
            "evidence_top_evidence_overlap": 0.3,
            "evidence_aligned_title_entity_match": audit.aligned_title_entity_match,
            "evidence_answer_year_in_prompt": audit.answer_year_in_prompt,
            "evidence_aligned_year_refuted_count": audit.aligned_year_refuted_count,
        }
    )

    score, reason, kind = overlay_decision(row, cap=0.72)
    assert kind == "year"
    assert reason == "aligned_year_refuted"
    assert score == 0.72


def test_v4_overlay_skips_answer_year_already_in_prompt() -> None:
    row = pd.Series(
        {
            "prompt": "В каком году после реформы 1853 года вышел указ?",
            "is_hallucination_proba": 0.2,
            "evidence_aligned_hit_count": 1,
            "evidence_top_evidence_overlap": 0.3,
            "evidence_aligned_title_entity_match": 1,
            "evidence_answer_year_in_prompt": 1,
            "evidence_aligned_year_refuted_count": 1,
        }
    )

    score, reason, _ = overlay_decision(row, cap=0.82)

    assert score == 0.2
    assert reason == "unchanged_answer_year_in_prompt"


def test_v4_count_overlay_requires_explicit_count_question() -> None:
    count_row = pd.Series(
        {
            "prompt": "Сколько спутников у Марса?",
            "is_hallucination_proba": 0.1,
            "evidence_aligned_hit_count": 1,
            "evidence_top_evidence_overlap": 0.3,
            "evidence_aligned_title_entity_match": 1,
            "evidence_answer_number_in_prompt": 0,
            "evidence_aligned_number_refuted_count": 1,
        }
    )
    non_count_row = count_row.copy()
    non_count_row["prompt"] = "Какая техника имеет 2 режима?"

    score, reason, kind = overlay_decision(count_row, cap=0.65)
    skipped_score, skipped_reason, skipped_kind = overlay_decision(non_count_row, cap=0.65)

    assert (score, reason, kind) == (0.65, "aligned_number_refuted", "count")
    assert skipped_score == 0.1
    assert skipped_reason == "unchanged_expected_none"
    assert skipped_kind == "none"


def test_v4_cap_variants_never_decrease_and_effect_report(tmp_path) -> None:
    scored = tmp_path / "scored.csv"
    pd.DataFrame(
        [
            {
                "prompt": "В каком году был основан Санкт-Петербург?",
                "model_answer": "В 1492 году.",
                "is_hallucination": 1,
                "is_hallucination_proba": 0.2,
                "evidence_aligned_hit_count": 1,
                "evidence_top_evidence_overlap": 0.3,
                "evidence_aligned_title_entity_match": 1,
                "evidence_answer_year_in_prompt": 0,
                "evidence_aligned_year_refuted_count": 1,
                "evidence_aligned_number_refuted_count": 0,
            },
            {
                "prompt": "В каком журнале в 1853 году был опубликован рассказ?",
                "model_answer": "В журнале Современник.",
                "is_hallucination": 0,
                "is_hallucination_proba": 0.3,
                "evidence_aligned_hit_count": 1,
                "evidence_top_evidence_overlap": 0.3,
                "evidence_aligned_title_entity_match": 1,
                "evidence_answer_year_in_prompt": 0,
                "evidence_aligned_year_refuted_count": 1,
                "evidence_aligned_number_refuted_count": 0,
            },
        ]
    ).to_csv(scored, index=False)
    summary = apply_overlay_v4(scored, tmp_path / "overlay", caps=(0.65, 0.82))
    cap65 = pd.read_csv(tmp_path / "overlay_cap065.csv")
    module_cap65 = apply_overlay_frame(pd.read_csv(scored), cap=0.65)

    assert summary["best"]["never_decreased"]
    assert cap65["is_hallucination_proba"].equals(module_cap65["is_hallucination_proba"])
    assert (cap65["refute_overlay_delta"] >= 0).all()
    assert cap65.loc[0, "is_hallucination_proba"] == 0.65
    assert cap65.loc[1, "is_hallucination_proba"] == 0.3

    report = analyze_overlay_effects(scored, tmp_path / "overlay_cap065.csv", tmp_path / "changed.csv")
    assert report["changed_rows"] == 1
    assert report["changed_precision"] == 1.0


def test_feature_extractor_appends_evidence_without_changing_text_only_shape() -> None:
    evidence = verify_against_evidence(
        "Who wrote Hamlet?",
        "William Shakespeare wrote Hamlet.",
        [],
    )
    extractor = FeatureExtractor()
    text = extractor.extract_text_only("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.")
    full = extractor.extract(
        "Who wrote Hamlet?",
        "William Shakespeare wrote Hamlet.",
        audit=AuditPayload.neutral(status="disabled", mode="dataset", model_name=None, ok=False),
        evidence_audit=evidence,
    )

    assert text.shape == (len(FeatureExtractor.text_feature_names),)
    assert full.shape == (
        len(FeatureExtractor.api_feature_names) + len(FeatureExtractor.text_feature_names) + len(FeatureExtractor.evidence_feature_names),
    )


def test_v5_entity_extraction_handles_russian_yo_normalization() -> None:
    entities = extract_core_entities("Пётр Первый", prompt="Кто основал город?", allow_prompt_entities=True)

    assert "петр первый" in entities


def test_v5_evidence_diagnostics_detect_aligned_alternative_entity() -> None:
    audit = verify_against_evidence(
        "Who wrote Hamlet?",
        "Christopher Marlowe.",
        [
            EvidenceSnippet(
                id=1,
                title="Hamlet",
                text="Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601.",
                source="wikipedia:en",
                score=1.0,
            )
        ],
    )

    assert audit.aligned_title_entity_match == 1.0
    assert audit.answer_core_entity_missing_from_aligned_evidence_count >= 1
    assert audit.aligned_alternative_entity_count >= 1
    assert audit.aligned_entity_confidence > 0.0


def test_v5_entity_overlay_triggers_and_never_decreases() -> None:
    compact = {
        "aligned_snippets": [
            {
                "title": "Hamlet",
                "text": "Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601.",
                "source": "wikipedia:en",
            }
        ]
    }
    frame = pd.DataFrame(
        [
            {
                "prompt": "Who wrote Hamlet?",
                "model_answer": "Christopher Marlowe.",
                "is_hallucination": 1,
                "is_hallucination_proba": 0.2,
                "evidence_status": "ok",
                "evidence_top_evidence_overlap": 0.4,
                "evidence_aligned_hit_count": 1,
                "evidence_aligned_title_entity_match": 1,
                "evidence_compact_json": json.dumps(compact),
            },
            {
                "prompt": "What is water?",
                "model_answer": "Water is H2O.",
                "is_hallucination": 0,
                "is_hallucination_proba": 0.7,
                "evidence_status": "ok",
                "evidence_top_evidence_overlap": 0.4,
                "evidence_aligned_hit_count": 1,
                "evidence_aligned_title_entity_match": 1,
                "evidence_compact_json": json.dumps(compact),
            },
        ]
    )

    out = apply_entity_overlay_frame(frame, cap=0.72)

    assert out.loc[0, "is_hallucination_proba"] == 0.72
    assert out.loc[0, "entity_overlay_v5_reason"] == "aligned_entity_mismatch"
    assert out.loc[1, "is_hallucination_proba"] == 0.7
    assert (out["entity_overlay_v5_delta"] >= 0).all()


def test_v5_entity_overlay_skips_long_numeric_supported_and_monotonic_caps() -> None:
    compact = {
        "aligned_snippets": [
            {
                "title": "Hamlet",
                "text": "Hamlet is a tragedy written by William Shakespeare.",
                "source": "wikipedia:en",
            }
        ]
    }
    base = {
        "evidence_status": "ok",
        "evidence_top_evidence_overlap": 0.4,
        "evidence_aligned_hit_count": 1,
        "evidence_aligned_title_entity_match": 1,
        "evidence_compact_json": json.dumps(compact),
    }
    frame = pd.DataFrame(
        [
            {"prompt": "Who wrote Hamlet?", "model_answer": "Christopher Marlowe " * 41, "is_hallucination_proba": 0.1, **base},
            {"prompt": "When was Hamlet written?", "model_answer": "Christopher Marlowe.", "is_hallucination_proba": 0.1, **base},
            {"prompt": "Who wrote Hamlet?", "model_answer": "William Shakespeare.", "is_hallucination_proba": 0.1, **base},
            {"prompt": "Who wrote Hamlet?", "model_answer": "Christopher Marlowe.", "is_hallucination_proba": 0.1, **base},
        ]
    )

    cap65 = apply_entity_overlay_frame(frame, cap=0.65)
    cap82 = apply_entity_overlay_frame(frame, cap=0.82)

    assert cap65.loc[0, "entity_overlay_v5_reason"] == "unchanged_long_answer"
    assert cap65.loc[1, "entity_overlay_v5_reason"] == "unchanged_expected_numeric"
    assert cap65.loc[2, "entity_overlay_v5_reason"] == "unchanged_answer_entity_supported"
    assert cap65.loc[3, "is_hallucination_proba"] == 0.65
    assert cap82.loc[3, "is_hallucination_proba"] == 0.82


def test_v6_answer_shape_computes_tail_fields() -> None:
    shape = answer_shape("Explain the topic", "Core answer is short. Extra unsupported sentence has Alice and Bob. Another tail sentence.")

    assert shape["answer_word_count"] > 8
    assert shape["answer_sentence_count"] == 3
    assert shape["answer_tail_word_count"] > 0
    assert 0 < shape["answer_tail_ratio"] < 1


def test_v6_drift_overlay_triggers_supported_core_unsupported_tail_and_never_decreases() -> None:
    long_answer = (
        "Photosynthesis is the process by which plants convert light into chemical energy. "
        "The answer also claims Alice Curie designed the first solar cathedral in 1901. "
        "It further says Bob Newton measured seven invisible pigments during the same expedition. "
        "These extra details are not part of the supported core and create answer drift."
    )
    frame = pd.DataFrame(
        [
            {
                "prompt": "Explain photosynthesis",
                "model_answer": long_answer,
                "is_hallucination_proba": 0.2,
                "score_path": "main",
                "audit_status": "ok",
                "evidence_status": "ok",
                "evidence_retrieval_hit_count": 4,
                "evidence_top_evidence_overlap": 0.2,
                "evidence_core_supported": 1,
                "evidence_core_refuted": 0,
                "evidence_claim_refuted_count": 0,
                "evidence_claim_unknown_count": 2,
                "evidence_tail_unsupported_entity_count": 2,
                "evidence_tail_unsupported_number_count": 1,
                "evidence_answer_entity_not_in_evidence_ratio": 0.8,
                "refute_overlay_delta": 0.0,
            }
        ]
    )

    out = apply_drift_overlay_frame(frame, cap=0.60)

    assert out.loc[0, "is_hallucination_proba"] == 0.60
    assert out.loc[0, "drift_overlay_v6_reason"] == "supported_core_unsupported_tail"
    assert out.loc[0, "drift_overlay_v6_delta"] >= 0


def test_v6_drift_overlay_skips_typed_numeric_v4_changed_and_supported_tail() -> None:
    long_answer = " ".join(["This is a long explanatory answer with enough words for the drift overlay."] * 5)
    base = {
        "model_answer": long_answer,
        "is_hallucination_proba": 0.2,
        "score_path": "fallback",
        "audit_status": "http_429",
        "evidence_status": "ok",
        "evidence_retrieval_hit_count": 4,
        "evidence_top_evidence_overlap": 0.05,
        "evidence_core_supported": 1,
        "evidence_core_refuted": 0,
        "evidence_claim_refuted_count": 0,
        "evidence_claim_unknown_count": 1,
        "evidence_tail_unsupported_entity_count": 2,
        "evidence_tail_unsupported_number_count": 0,
        "evidence_answer_entity_not_in_evidence_ratio": 0.8,
    }
    frame = pd.DataFrame(
        [
            {"prompt": "Who wrote Hamlet?", "refute_overlay_delta": 0.0, **base},
            {"prompt": "When was Hamlet written?", "refute_overlay_delta": 0.0, **base},
            {"prompt": "Explain Hamlet", "refute_overlay_delta": 0.1, **base},
            {
                "prompt": "Explain Hamlet",
                "refute_overlay_delta": 0.0,
                **{**base, "evidence_claim_unknown_count": 0, "evidence_tail_unsupported_entity_count": 0, "evidence_answer_entity_not_in_evidence_ratio": 0.0},
            },
        ]
    )

    out = apply_drift_overlay_frame(frame, cap=0.55)

    assert out.loc[0, "drift_overlay_v6_reason"] == "unchanged_profile_not_allowed"
    assert out.loc[1, "drift_overlay_v6_reason"] == "unchanged_profile_not_allowed"
    assert out.loc[2, "drift_overlay_v6_reason"] == "unchanged_v4_already_changed"
    assert out.loc[3, "drift_overlay_v6_reason"] == "unchanged_no_drift_trigger"


def test_v6_drift_overlay_script_tiny_csv_and_effect_reason(tmp_path) -> None:
    long_answer = (
        "Photosynthesis is the process by which plants convert light into chemical energy. "
        "Alice Curie designed the first solar cathedral in 1901. "
        "Bob Newton measured seven invisible pigments during the same expedition. "
        "Those extra claims are unsupported by the evidence and make the answer drift. "
        "The tail continues with another invented institution, a hidden laboratory, and several unsupported historical details."
    )
    scored = tmp_path / "scored.csv"
    pd.DataFrame(
        [
            {
                "prompt": "Explain photosynthesis",
                "model_answer": long_answer,
                "is_hallucination": 1,
                "is_hallucination_proba": 0.2,
                "score_path": "main",
                "audit_status": "ok",
                "evidence_status": "ok",
                "evidence_retrieval_hit_count": 4,
                "evidence_top_evidence_overlap": 0.2,
                "evidence_core_supported": 1,
                "evidence_core_refuted": 0,
                "evidence_claim_refuted_count": 0,
                "evidence_claim_unknown_count": 2,
                "evidence_tail_unsupported_entity_count": 2,
                "evidence_tail_unsupported_number_count": 1,
                "evidence_answer_entity_not_in_evidence_ratio": 0.8,
                "refute_overlay_delta": 0.0,
            },
            {
                "prompt": "What is water?",
                "model_answer": "Water is H2O.",
                "is_hallucination": 0,
                "is_hallucination_proba": 0.1,
                "score_path": "main",
                "audit_status": "ok",
                "evidence_status": "ok",
            },
        ]
    ).to_csv(scored, index=False)

    summary = apply_drift_overlay_v6(scored, tmp_path / "v6", caps=(0.55,))
    report = analyze_overlay_effects(scored, tmp_path / "v6_cap055.csv", tmp_path / "changed.csv")

    assert summary["reports"][0]["changed_rows"] == 1
    assert "drift_overlay_v6_reason" in pd.read_csv(tmp_path / "v6_cap055.csv").columns
    assert report["changed_by_overlay_reason"] == {"supported_core_unsupported_tail": 1}
