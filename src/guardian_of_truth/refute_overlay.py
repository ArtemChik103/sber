from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from guardian_of_truth.cache import SQLiteCache
from guardian_of_truth.utils import DATA_DIR, MODEL_DIR, sha256_hexdigest


REFUTE_OVERLAY_POLICY = "v4_cap082"
REFUTE_OVERLAY_CAP = 0.82
DISABLED_REFUTE_OVERLAY_POLICY = "disabled"
BAYESIAN_BLIND_RESCUE_POLICY = "v5_bayesian_blind_rescue"
SUPPORTED_REFUTE_OVERLAY_POLICIES = {
    DISABLED_REFUTE_OVERLAY_POLICY,
    REFUTE_OVERLAY_POLICY,
    BAYESIAN_BLIND_RESCUE_POLICY,
}
MIN_STRONG_TOP_OVERLAP = 0.28
DEFAULT_BLIND_CACHE_PATH = DATA_DIR / "cache" / "groq_cache.sqlite"

YEAR_PATTERNS = (
    r"\bв\s+каком\s+году\b",
    r"\bкакого\s+года\b",
    r"\bкогда\b",
    r"\bwhen\b",
    r"\bwhat\s+year\b",
)
DATE_PATTERNS = (
    r"\bв\s+каком\s+месяце\b",
    r"\bкакая\s+дата\b",
    r"\bкакого\s+числа\b",
)
COUNT_PATTERNS = (
    r"\bсколько\b",
    r"\bhow\s+many\b",
    r"\bhow\s+much\b",
)


def expected_answer_kind(prompt: str) -> str:
    lower = str(prompt or "").lower()
    if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in YEAR_PATTERNS):
        return "year"
    if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in DATE_PATTERNS):
        return "date_or_month"
    if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in COUNT_PATTERNS):
        return "count"
    return "none"


def _num(row: Any, column: str) -> float:
    value = row.get(column, 0.0)
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        return 0.0
    return float(value)


def overlay_decision(row: Any, *, cap: float = REFUTE_OVERLAY_CAP) -> tuple[float, str, str]:
    base = _num(row, "is_hallucination_proba")
    kind = expected_answer_kind(str(row.get("prompt", "")))
    if kind == "none":
        return base, "unchanged_expected_none", kind
    if _num(row, "evidence_aligned_hit_count") < 1:
        return base, "unchanged_no_aligned_hits", kind
    if _num(row, "evidence_top_evidence_overlap") < MIN_STRONG_TOP_OVERLAP:
        return base, "unchanged_low_strong_overlap", kind

    if kind in {"year", "date_or_month"}:
        if _num(row, "evidence_answer_year_in_prompt") > 0:
            return base, "unchanged_answer_year_in_prompt", kind
        if _num(row, "evidence_aligned_year_refuted_count") > 0:
            return max(base, cap), "aligned_year_refuted", kind
        return base, "unchanged_no_aligned_year_refute", kind

    if kind == "count":
        if _num(row, "evidence_answer_number_in_prompt") > 0:
            return base, "unchanged_answer_number_in_prompt", kind
        if _num(row, "evidence_aligned_number_refuted_count") > 0:
            return max(base, cap), "aligned_number_refuted", kind
        return base, "unchanged_no_aligned_number_refute", kind

    return base, "unchanged_expected_none", kind


def row_from_evidence_audit(prompt: str, answer: str, base_score: float, evidence_audit: Any | None) -> dict[str, Any]:
    row = {
        "prompt": prompt,
        "model_answer": answer,
        "is_hallucination_proba": float(base_score),
    }
    if evidence_audit is None:
        return row
    row.update(
        {
            "evidence_aligned_hit_count": evidence_audit.aligned_hit_count,
            "evidence_aligned_year_refuted_count": evidence_audit.aligned_year_refuted_count,
            "evidence_aligned_number_refuted_count": evidence_audit.aligned_number_refuted_count,
            "evidence_answer_year_in_prompt": evidence_audit.answer_year_in_prompt,
            "evidence_answer_number_in_prompt": evidence_audit.answer_number_in_prompt,
            "evidence_top_evidence_overlap": evidence_audit.top_evidence_overlap,
        }
    )
    return row


RUSSIAN_ENDINGS = (
    "ский", "ского", "скому", "ским", "ском", "ская", "скую", "ской", "ские", "ских",
    "ели", "еля", "елю", "елем", "еле", "ова", "ову", "овым", "ове", "овы", "ов", "ев",
    "ем", "ом", "ам", "ах", "ях", "ии", "ию", "ия", "ей", "ой", "ай", "ый", "ий",
    "ая", "ое", "ее", "ые", "ие", "ль", "ле", "лю", "ля", "рь", "ре", "рю", "ря",
    "а", "е", "и", "о", "у", "ы", "ь", "я",
)


def _stem_ru(w: str) -> str:
    w = w.lower()
    for end in RUSSIAN_ENDINGS:
        if len(w) > len(end) + 3 and w.endswith(end):
            return w[:-len(end)]
    return w


def _matches_blind_fact(fact: str, candidate: str) -> bool:
    fact_lower = fact.lower()
    cand_lower = candidate.lower()

    fact_years = re.findall(r"\b(1\d{3}|20\d{2})\b", fact_lower)
    cand_years = re.findall(r"\b(1\d{3}|20\d{2})\b", cand_lower)
    if fact_years and cand_years and any(y in cand_years for y in fact_years):
        return True

    fact_words = [
        w for w in re.findall(r"[а-яёa-z0-9]+", fact_lower)
        if len(w) >= 3 and w not in {"года", "году", "были", "была", "было", "это", "был"}
    ]
    if not fact_words:
        return False

    cand_words = [w for w in re.findall(r"[а-яёa-z0-9]+", cand_lower) if len(w) >= 3]
    cand_stems = {_stem_ru(w) for w in cand_words}

    for fw in fact_words:
        fst = _stem_ru(fw)
        if len(fst) >= 3 and any(fst in cs or cs.startswith(fst[:4]) for cs in cand_stems):
            return True
        if fw in cand_lower:
            return True
    return False



def _compute_year_delta(
    prompt: str,
    answer: str,
    evidence_audit: Any | None,
    cache: SQLiteCache | None = None,
) -> float | None:
    prompt_years = set(re.findall(r"\b(1\d{3}|20\d{2})\b", prompt))
    cand_years = [int(x) for x in re.findall(r"\b(1\d{3}|20\d{2})\b", answer) if x not in prompt_years]
    if not cand_years:
        return None

    # 1. Check blind fact year from cache
    if cache is not None:
        ckey = sha256_hexdigest("blind-fact-v1", prompt)
        bf = cache.get(ckey)
        if bf is not None and "blind_fact" in bf:
            bf_years = [int(x) for x in re.findall(r"\b(1\d{3}|20\d{2})\b", str(bf["blind_fact"])) if x not in prompt_years]
            if bf_years:
                delta = min(abs(cy - by) for cy in cand_years for by in bf_years)
                if delta > 2:
                    return float(delta)

    # 2. Check evidence audit snippets
    if evidence_audit is None:
        return None

    prompt_sub = [
        w for w in re.findall(r"[а-яёa-z0-9]+", prompt.lower())
        if len(w) >= 3 and w not in {"году", "года", "когда", "каком", "какой", "какая", "были", "была", "было", "был", "это"}
    ]
    if not prompt_sub:
        return None
    prompt_stems = {_stem_ru(w) for w in prompt_sub}

    cj_str = getattr(evidence_audit, "compact_json", None)
    if not cj_str:
        return None
    try:
        cj = json.loads(cj_str)
    except Exception:
        return None

    snippets = cj.get("snippets", []) + cj.get("overlay_eligible_snippets", [])
    for sn in snippets:
        stext = str(sn.get("text", "")) + " " + str(sn.get("title", ""))
        swords = [w for w in re.findall(r"[а-яёa-z0-9]+", stext.lower()) if len(w) >= 3]
        sstems = {_stem_ru(w) for w in swords}

        matched_stems = sum(1 for pst in prompt_stems if any(pst in ss or ss.startswith(pst[:4]) for ss in sstems))
        match_ratio = matched_stems / len(prompt_stems)

        if match_ratio >= 0.45 and matched_stems >= 2:
            sn_years = [int(x) for x in re.findall(r"\b(1\d{3}|20\d{2})\b", stext) if x not in prompt_years]
            if sn_years:
                min_delta = min(abs(cy - sy) for cy in cand_years for sy in sn_years)
                return float(min_delta)
    return None


def _arbitrate_with_gpt120b(prompt: str, answer: str, cache: SQLiteCache | None) -> tuple[bool | None, float]:
    if cache is None:
        return None, 0.0
    ckey = sha256_hexdigest("gpt120b-arbitration-v1", prompt, answer)
    entry = cache.get(ckey)
    if entry is not None and "is_hallucination" in entry:
        is_h = bool(entry["is_hallucination"])
        conf = float(entry.get("confidence", 0.90))
        return is_h, conf
    return None, 0.0


def bayesian_decision(
    prompt: str,
    answer: str,
    base_score: float,
    evidence_audit: Any | None,
    *,
    rescue_thresh: float = 0.65,
    rescue_delta: float = 0.50,
    rescue_floor: float = 0.12,
    boost_overlap: float = 0.25,
    boost_delta: float = 0.40,
    cache: SQLiteCache | None = None,
) -> tuple[float, str, str]:
    kind = expected_answer_kind(prompt)
    curr_score = float(base_score)

    # 1. Blind recall positive rescue with Russian morphological stemming (resolves False Positives)
    target_cache = cache
    if target_cache is None and DEFAULT_BLIND_CACHE_PATH.exists():
        try:
            target_cache = SQLiteCache(DEFAULT_BLIND_CACHE_PATH)
        except Exception:
            target_cache = None

    if target_cache is not None:
        ckey = sha256_hexdigest("blind-fact-v1", prompt)
        entry = target_cache.get(ckey)
        if entry is not None and "blind_fact" in entry:
            fact = str(entry["blind_fact"]).strip()
            if _matches_blind_fact(fact, answer):
                if curr_score > rescue_thresh:
                    rescued_score = float(max(rescue_floor, curr_score - rescue_delta))
                    return rescued_score, "blind_recall_rescue_fp", kind

    # 2. Continuous Year Delta refutation (resolves False Negatives for when/year questions)
    if kind in {"year", "date_or_month"}:
        year_delta = _compute_year_delta(prompt, answer, evidence_audit, cache=target_cache)
        if year_delta is not None and year_delta > 2 and curr_score < 0.50:
            boosted_score = float(min(0.85, curr_score + 0.40))
            return boosted_score, "evidence_year_delta_boost_fn", kind

    # 3. Evidence Refutation boost (resolves False Negatives)
    if evidence_audit is not None:
        ref_y = getattr(evidence_audit, "aligned_year_refuted_count", 0.0) > 0
        ref_n = getattr(evidence_audit, "aligned_number_refuted_count", 0.0) > 0
        ref_c = getattr(evidence_audit, "core_refuted", 0.0) > 0
        overlap = getattr(evidence_audit, "top_evidence_overlap", 0.0)
        if (ref_y or ref_c or (ref_n and overlap >= 0.35)) and overlap >= boost_overlap and curr_score < 0.50:
            boosted_score = float(min(0.85, curr_score + boost_delta))
            return boosted_score, "evidence_refuted_boost_fn", kind

    # 4. Dual-Model Arbitration with openai/gpt-oss-120b for high-uncertainty gray zone
    if 0.38 <= curr_score <= 0.62 and target_cache is not None:
        is_h, conf = _arbitrate_with_gpt120b(prompt, answer, target_cache)
        if is_h is not None and conf >= 0.95:
            if is_h:
                curr_score = float(min(0.85, curr_score + 0.02))
            else:
                curr_score = float(max(0.15, curr_score - 0.02))
            return curr_score, "gpt120b_arbitration", kind

    return curr_score, "unchanged", kind


def apply_policy_to_score(
    prompt: str,
    answer: str,
    base_score: float,
    evidence_audit: Any | None,
    *,
    policy: str = DISABLED_REFUTE_OVERLAY_POLICY,
    cache: SQLiteCache | None = None,
) -> tuple[float, str, str]:
    if policy == DISABLED_REFUTE_OVERLAY_POLICY:
        return float(base_score), "disabled", "none"
    if policy == BAYESIAN_BLIND_RESCUE_POLICY:
        return bayesian_decision(prompt, answer, base_score, evidence_audit, cache=cache)
    if policy == REFUTE_OVERLAY_POLICY:
        row = row_from_evidence_audit(prompt, answer, base_score, evidence_audit)
        return overlay_decision(row, cap=REFUTE_OVERLAY_CAP)
    raise ValueError(f"Unsupported refute overlay policy: {policy}")


def apply_overlay_frame(
    frame: pd.DataFrame,
    *,
    policy: str = REFUTE_OVERLAY_POLICY,
    cap: float = REFUTE_OVERLAY_CAP,
    cache: SQLiteCache | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    base = output["is_hallucination_proba"].astype(float).copy()
    if policy == REFUTE_OVERLAY_POLICY:
        decisions = output.apply(lambda row: overlay_decision(row, cap=cap), axis=1)
        output["base_is_hallucination_proba"] = base
        output["is_hallucination_proba"] = [score for score, _, _ in decisions]
        output["refute_overlay_v4"] = [reason for _, reason, _ in decisions]
        output["refute_overlay_v4_expected_kind"] = [kind for _, _, kind in decisions]
        output["refute_overlay_v4_cap"] = cap
        output["refute_overlay_delta"] = output["is_hallucination_proba"] - output["base_is_hallucination_proba"]
        return output

    decisions = [
        apply_policy_to_score(
            str(row.get("prompt", "")),
            str(row.get("model_answer", "")),
            float(row.get("is_hallucination_proba", 0.0)),
            None,
            policy=policy,
            cache=cache,
        )
        for _, row in output.iterrows()
    ]
    output["base_is_hallucination_proba"] = base
    output["is_hallucination_proba"] = [score for score, _, _ in decisions]
    output["refute_overlay_policy"] = policy
    output["refute_overlay_reason"] = [reason for _, reason, _ in decisions]
    output["refute_overlay_expected_kind"] = [kind for _, _, kind in decisions]
    output["refute_overlay_delta"] = output["is_hallucination_proba"] - output["base_is_hallucination_proba"]
    return output


DEFAULT_ISOTONIC_CALIBRATOR_PATH = MODEL_DIR / "isotonic_calibrator.joblib"


def get_profile_shift(prompt: str) -> float:
    p = str(prompt or "").lower()
    if "кто" in p or "who" in p:
        return -0.10
    if any(k in p for k in ("какой", "какая", "какое", "какие", "what", "which")):
        return 0.10
    return 0.0


class IsotonicRankCalibrator:
    def __init__(self, path: Path | str = DEFAULT_ISOTONIC_CALIBRATOR_PATH) -> None:
        self.path = Path(path)
        self.model: Any | None = None
        if self.path.exists():
            try:
                self.model = joblib.load(self.path)
            except Exception:
                self.model = None

    def fit(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        prompts: list[str] | None = None,
        save: bool = True,
    ) -> "IsotonicRankCalibrator":
        from scipy.special import expit, logit
        from sklearn.isotonic import IsotonicRegression

        arr = np.asarray(scores, dtype=float).copy()
        if prompts is not None and len(prompts) == len(arr):
            eps = 1e-4
            shifts = np.array([get_profile_shift(p) for p in prompts])
            logits = logit(np.clip(arr, eps, 1.0 - eps)) + shifts
            arr = expit(logits)

        iso = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
        iso.fit(arr, y_true)
        self.model = iso
        if save:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.path)
        return self

    def predict(
        self,
        scores: np.ndarray | list[float] | float,
        prompts: list[str] | str | None = None,
    ) -> np.ndarray | float:
        if self.model is None:
            return np.asarray(scores) if not isinstance(scores, (float, int)) else float(scores)
        from scipy.special import expit, logit

        arr = np.asarray(scores, dtype=float)
        single = (arr.ndim == 0)
        flat = arr.flatten().copy()

        if prompts is not None:
            eps = 1e-4
            if isinstance(prompts, str):
                p_list = [prompts]
            else:
                p_list = list(prompts)
            if len(p_list) == len(flat):
                shifts = np.array([get_profile_shift(p) for p in p_list])
                logits = logit(np.clip(flat, eps, 1.0 - eps)) + shifts
                flat = expit(logits)

        calibrated = self.model.predict(flat)
        if single:
            return float(calibrated[0])
        return calibrated.reshape(arr.shape)


def apply_isotonic_calibration(
    scores: np.ndarray | list[float],
    prompts: list[str] | str | None = None,
    calibrator_path: Path | str = DEFAULT_ISOTONIC_CALIBRATOR_PATH,
) -> np.ndarray:
    calibrator = IsotonicRankCalibrator(calibrator_path)
    return np.asarray(calibrator.predict(scores, prompts=prompts), dtype=float)


