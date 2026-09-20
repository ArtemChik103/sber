from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from guardian_of_truth.guardian import GuardianOfTruth, ScoringResult
from guardian_of_truth.refute_overlay import (
    BAYESIAN_BLIND_RESCUE_POLICY,
    DISABLED_REFUTE_OVERLAY_POLICY,
    REFUTE_OVERLAY_POLICY,
    SUPPORTED_REFUTE_OVERLAY_POLICIES,
)
from guardian_of_truth.utils import MODEL_DIR, load_local_env

EXAMPLES: list[dict[str, str]] = [
    {
        "title": "Основание Петербурга",
        "label": "Корректный исторический факт",
        "prompt": "В каком году был основан Санкт-Петербург?",
        "answer": "Санкт-Петербург был основан Петром I в 1703 году.",
    },
    {
        "title": "Первый президент США",
        "label": "Грубая хронологическая ошибка",
        "prompt": "Кто был первым президентом США?",
        "answer": "Первым президентом США был Авраам Линкольн в 1861 году.",
    },
    {
        "title": "Полет Гагарина",
        "label": "Тонкое числовое искажение даты",
        "prompt": "В каком году Юрий Гагарин совершил первый в истории человечества полет в космос?",
        "answer": "Юрий Гагарин совершил первый космический полет на корабле «Восток-1» 12 апреля 1963 года.",
    },
    {
        "title": "Столица Австралии",
        "label": "Распространенная фактологическая путаница",
        "prompt": "Какова официальная столица Австралии?",
        "answer": "Официальной столицей Австралии является Сидней, крупнейший экономический центр страны.",
    },
]

CSS_STYLES = """
<style>
    /* Clean devtools typography and layout */
    .stApp {
        font-feature-settings: "cv02", "cv03", "cv04", "cv11";
    }
    .metric-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 1.4rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.75rem;
        color: rgba(128, 128, 128, 0.9);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .verdict-banner {
        border-radius: 6px;
        padding: 16px 20px;
        margin: 16px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .verdict-banner-factual {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #10b981;
    }
    .verdict-banner-hallucination {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #ef4444;
    }
    .verdict-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
    }
    .verdict-sub {
        font-size: 0.85rem;
        margin-top: 2px;
        opacity: 0.85;
    }
    .verdict-score {
        font-family: ui-monospace, monospace;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: right;
    }
</style>
"""


def resolve_groq_api_key() -> str | None:
    load_local_env()
    try:
        if "GROQ_API_KEY" in st.secrets:
            key = str(st.secrets["GROQ_API_KEY"]).strip()
            if key:
                os.environ["GROQ_API_KEY"] = key
                return key
    except Exception:
        pass

    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    return env_key if env_key else None


@st.cache_resource(show_spinner=False)
def get_cached_guardian(
    model_dir: str = "model",
    policy: str = BAYESIAN_BLIND_RESCUE_POLICY,
) -> GuardianOfTruth:
    return GuardianOfTruth(
        model_dir=model_dir,
        refute_overlay_policy=policy,
    )


def format_scoring_payload(engine: GuardianOfTruth, result: ScoringResult) -> dict[str, Any]:
    return {
        "verdict": "Hallucination" if result.is_hallucination else "Likely factual",
        "is_hallucination": result.is_hallucination,
        "is_hallucination_proba": round(result.is_hallucination_proba, 6),
        "timing": {
            "t_total_ms": round(result.t_total_sec * 1000, 1),
            "t_model_ms": round(result.t_model_sec * 1000, 1),
            "t_overhead_ms": round(result.t_overhead_sec * 1000, 1),
        },
        "pipeline": {
            "score_path": getattr(engine, "last_score_path", "unknown"),
            "base_proba": round(getattr(engine, "last_base_is_hallucination_proba", result.is_hallucination_proba) or 0.0, 6),
            "overlay_policy": getattr(engine, "last_refute_overlay_policy", "none"),
            "overlay_reason": getattr(engine, "last_refute_overlay_reason", "none"),
            "overlay_kind": getattr(engine, "last_refute_overlay_expected_kind", "none"),
            "overlay_delta": round(getattr(engine, "last_refute_overlay_delta", 0.0), 6),
        },
    }


def render_app() -> None:
    st.set_page_config(
        page_title="Guardian of Truth",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

    detected_key = resolve_groq_api_key()

    with st.sidebar:
        st.title("Guardian of Truth")
        st.caption("Factual Hallucination Detector v9.4")

        st.subheader("Параметры окружения")
        if detected_key:
            st.success("Groq API: подключен")
        else:
            st.warning("Groq API: ключ не обнаружен")
            user_key = st.text_input(
                "GROQ_API_KEY",
                type="password",
                placeholder="gsk_...",
                help="Введите ключ Groq API для активации режима реального времени",
            )
            if user_key.strip():
                os.environ["GROQ_API_KEY"] = user_key.strip()
                st.rerun()

        st.subheader("Политика верификации")
        selected_policy = st.selectbox(
            "Алгоритм оверлея",
            options=SUPPORTED_REFUTE_OVERLAY_POLICIES,
            index=SUPPORTED_REFUTE_OVERLAY_POLICIES.index(BAYESIAN_BLIND_RESCUE_POLICY)
            if BAYESIAN_BLIND_RESCUE_POLICY in SUPPORTED_REFUTE_OVERLAY_POLICIES
            else 0,
            format_func=lambda x: {
                BAYESIAN_BLIND_RESCUE_POLICY: "Bayesian Blind Rescue (v9.4, Рекомендуется)",
                REFUTE_OVERLAY_POLICY: "Evidence-Grounded Refute (v8)",
                DISABLED_REFUTE_OVERLAY_POLICY: "Disabled (Базовый ансамбль)",
            }.get(x, x),
        )

        st.divider()
        st.subheader("Метрики валидации")
        st.markdown(
            """
            - **PR-AUC**: `0.9262` (Рекорд v9.4)
            - **ROC-AUC**: `0.9187`
            - **Задержка**: `< 200 мс` (при SLA 500 мс)
            - **Память**: `~180 МБ RAM`
            """
        )

    st.header("Аудит фактологической достоверности")
    st.markdown("Детекция фактологических галлюцинаций в генерациях LLM на основе гибридного ансамбля.")

    st.subheader("Контрольные примеры")
    example_cols = st.columns(len(EXAMPLES))
    for idx, (col, ex) in enumerate(zip(example_cols, EXAMPLES)):
        with col:
            if st.button(ex["title"], key=f"btn_ex_{idx}", use_container_width=True):
                st.session_state["prompt_val"] = ex["prompt"]
                st.session_state["answer_val"] = ex["answer"]
                st.session_state["example_label"] = ex["label"]

    current_prompt = st.session_state.get("prompt_val", EXAMPLES[0]["prompt"])
    current_answer = st.session_state.get("answer_val", EXAMPLES[0]["answer"])

    if "example_label" in st.session_state:
        st.caption(f"Выбран сценарий: {st.session_state['example_label']}")

    with st.form("audit_form"):
        prompt_input = st.text_area(
            "Вопрос или промпт:",
            value=current_prompt,
            height=100,
            placeholder="Введите вопрос пользователя...",
        )
        answer_input = st.text_area(
            "Кандидатный ответ модели:",
            value=current_answer,
            height=130,
            placeholder="Введите ответ модели для верификации...",
        )

        btn_cols = st.columns([1, 1, 6])
        with btn_cols[0]:
            submitted = st.form_submit_button("Проверить", type="primary", use_container_width=True)
        with btn_cols[1]:
            cleared = st.form_submit_button("Очистить", use_container_width=True)

    if cleared:
        st.session_state["prompt_val"] = ""
        st.session_state["answer_val"] = ""
        st.session_state.pop("example_label", None)
        st.rerun()

    if submitted:
        p_clean = prompt_input.strip()
        a_clean = answer_input.strip()

        if not p_clean or not a_clean:
            st.error("Для проверки необходимо заполнить оба поля: вопрос и ответ.")
            return

        engine = get_cached_guardian(model_dir=str(MODEL_DIR), policy=selected_policy)

        with st.spinner("Анализ фактологической согласованности..."):
            try:
                res = engine.score(p_clean, a_clean)
                payload = format_scoring_payload(engine, res)
            except Exception as exc:
                st.error(f"Ошибка выполнения инференса: {exc}")
                return

        prob = res.is_hallucination_proba
        is_hallucination = res.is_hallucination

        # Verdict banner
        if is_hallucination:
            st.markdown(
                f"""
                <div class="verdict-banner verdict-banner-hallucination">
                    <div>
                        <div class="verdict-title">Фактическая галлюцинация</div>
                        <div class="verdict-sub">В ответе обнаружены искажения дат, сущностей или противоречие проверенным фактам.</div>
                    </div>
                    <div class="verdict-score">{prob:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-banner verdict-banner-factual">
                    <div>
                        <div class="verdict-title">Достоверный ответ</div>
                        <div class="verdict-sub">Ответ согласуется с фактологической базой и не содержит противоречий.</div>
                    </div>
                    <div class="verdict-score">{prob:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Probability Progress Bar
        st.progress(float(prob))
        st.caption(f"Вероятность галлюцинации: {prob:.4f} (порог решения: 0.5000)")

        # Metric cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{res.t_total_sec * 1000:.1f} мс</div>
                    <div class="metric-label">Общая задержка</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{res.t_model_sec * 1000:.1f} мс</div>
                    <div class="metric-label">Время верификатора</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            path_name = "Основной контур" if payload["pipeline"]["score_path"] == "main" else "Оффлайн-фоллбек"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{path_name}</div>
                    <div class="metric-label">Трасса инференса</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col4:
            delta = payload["pipeline"]["overlay_delta"]
            sign = "+" if delta > 0 else ""
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{sign}{delta:.3f}</div>
                    <div class="metric-label">Коррекция оверлея</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Detail tabs
        tab1, tab2 = st.tabs(["Диагностика контура", "Телеметрия (JSON)"])
        with tab1:
            st.markdown("#### Параметры решения")
            st.write(
                {
                    "Правило оверлея": payload["pipeline"]["overlay_reason"],
                    "Целевая сущность": payload["pipeline"]["overlay_kind"],
                    "Базовая вероятность (до оверлея)": payload["pipeline"]["base_proba"],
                    "Скорректированная вероятность": payload["is_hallucination_proba"],
                }
            )
        with tab2:
            st.json(payload)


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
