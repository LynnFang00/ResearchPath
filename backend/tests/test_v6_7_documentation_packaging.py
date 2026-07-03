import inspect
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.services.ltr_model_service import V64_METHOD_NAME  # noqa: E402
from app.services.recommendation_service import SUPPORTED_RETRIEVAL_METHODS, recommend_from_query  # noqa: E402
from scripts.validate_v3_5_unjudged_audit_labels import (  # noqa: E402
    DEFAULT_SELECTED_240,
    EXPECTED_SELECTED_240_SHA256,
    EXPECTED_V21_LABELS_SHA256,
    EXPECTED_V25_LABELS_SHA256,
    EXPECTED_V32_LABELS_SHA256,
    file_sha256,
)
from scripts.evaluate_v3_6_revealed_v3_5_labels import DEFAULT_V35_LABELS, EXPECTED_V35_LABELS_SHA256  # noqa: E402


REPORT = ROOT / "data" / "eval" / "results" / "v6_7_documentation_packaging_report.json"
PORTFOLIO_DOC = ROOT / "docs" / "v6_7_portfolio_demo_packaging.md"
PROJECT_CONTEXT = ROOT / "PROJECT_CONTEXT.md"
README = ROOT / "README.md"
SEARCH_PAGE = ROOT / "frontend" / "src" / "pages" / "SearchPage.tsx"
CLIENT = ROOT / "frontend" / "src" / "api" / "client.ts"
V21_LABELS = ROOT / "data" / "eval" / "manual_labels_v2_1.jsonl"
V25_LABELS = ROOT / "data" / "eval" / "manual_labels_v2_5_hard_negatives.jsonl"
V32_LABELS = ROOT / "data" / "eval" / "manual_labels_v3_2_800.jsonl"
V39_LABELS = ROOT / "data" / "eval" / "manual_labels_v3_9_semantic_expansion.jsonl"


def test_v6_7_packaging_report_schema_and_scope() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["schema_version"] == "v6.7_documentation_packaging_report"
    assert report["runtime"]["backend_default_method"] == "bm25"
    assert report["runtime"]["safe_fusion_runtime_method"] == V64_METHOD_NAME
    assert report["frontend_demo"]["comparison_enabled_by_default"] is False
    assert report["scope"]["trained_new_models"] is False
    assert report["scope"]["added_or_edited_labels"] is False
    assert report["scope"]["expanded_corpus"] is False
    assert report["scope"]["changed_backend_default"] is False
    assert report["scope"]["changed_frontend_default"] is False
    assert report["scope"]["committed_or_pushed"] is False


def test_v6_7_docs_summarize_pipeline_and_safe_fusion() -> None:
    context = PROJECT_CONTEXT.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    portfolio = PORTFOLIO_DOC.read_text(encoding="utf-8")

    for text in (context, readme, portfolio):
        assert "V6.6" in text
        assert "v6_4_safe_fusion" in text
        assert "bm25" in text

    assert "Runtime formula parity max delta: `0.0`" in portfolio
    assert "Do not train new models" in context


def test_v6_7_frontend_demo_support_is_opt_in_and_defaults_stay_put() -> None:
    search_page = SEARCH_PAGE.read_text(encoding="utf-8")
    client = CLIENT.read_text(encoding="utf-8")

    assert "useState('embedding')" in search_page
    assert "const [compareMethods, setCompareMethods] = useState(false)" in search_page
    assert "v6_4_safe_fusion" in search_page
    assert "v4_9_guarded_text_blend" in search_page
    assert "compareRecommendationMethods" in search_page
    assert "fetchRecommendationMethods" in search_page

    assert "method = 'hybrid'" in client
    assert "fetchRecommendationMethods" in client
    assert "compareRecommendationMethods" in client


def test_v6_7_backend_default_and_opt_in_method_status() -> None:
    assert inspect.signature(recommend_from_query).parameters["method"].default == "bm25"
    assert V64_METHOD_NAME in SUPPORTED_RETRIEVAL_METHODS


def test_v6_7_protected_hashes_are_still_unchanged() -> None:
    v66_report = json.loads(
        (ROOT / "data" / "eval" / "results" / "v6_6_safe_fusion_scorer_reproduction_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert file_sha256(V21_LABELS) == EXPECTED_V21_LABELS_SHA256
    assert file_sha256(V25_LABELS) == EXPECTED_V25_LABELS_SHA256
    assert file_sha256(V32_LABELS) == EXPECTED_V32_LABELS_SHA256
    assert file_sha256(DEFAULT_V35_LABELS) == EXPECTED_V35_LABELS_SHA256
    assert file_sha256(DEFAULT_SELECTED_240) == EXPECTED_SELECTED_240_SHA256
    assert file_sha256(V39_LABELS) == v66_report["protected_hashes"]["v3_9_labels_sha256"]
