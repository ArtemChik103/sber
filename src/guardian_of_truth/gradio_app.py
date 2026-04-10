from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Any

import gradio as gr

from guardian_of_truth.guardian import GuardianOfTruth, ScoringResult


EXAMPLES = [
    [
        "В каком году был основан Санкт-Петербург?",
        "Санкт-Петербург был основан в 1703 году.",
    ],
    [
        "Кто был первым президентом США?",
        "Первым президентом США был Авраам Линкольн.",
    ],
]


@lru_cache(maxsize=4)
def get_guardian(model_dir: str = "model") -> GuardianOfTruth:
    return GuardianOfTruth(model_dir=model_dir)


def format_result(result: ScoringResult) -> tuple[str, dict[str, Any]]:
    verdict = "Hallucination" if result.is_hallucination else "Likely factual"
    payload = {
        "is_hallucination": result.is_hallucination,
        "predict_proba": round(result.is_hallucination_proba, 6),
        "t_model_sec": round(result.t_model_sec, 6),
        "t_overhead_sec": round(result.t_overhead_sec, 6),
        "t_total_sec": round(result.t_total_sec, 6),
    }
    return verdict, payload


def run_score(
    prompt: str,
    answer: str,
    *,
    guardian: GuardianOfTruth | None = None,
    model_dir: str = "model",
) -> tuple[str, dict[str, Any]]:
    prompt_clean = prompt.strip()
    answer_clean = answer.strip()
    if not prompt_clean or not answer_clean:
        return (
            "Input required",
            {"error": "Both prompt and answer must be non-empty strings."},
        )

    engine = guardian or get_guardian(model_dir)
    try:
        result = engine.score(prompt_clean, answer_clean)
    except Exception as exc:  # pragma: no cover
        return "Runtime error", {"error": str(exc)}
    return format_result(result)


def build_demo(*, model_dir: str = "model") -> gr.Blocks:
    with gr.Blocks(title="Guardian of Truth") as demo:
        gr.Markdown(
            """
            # Guardian of Truth
            Minimal UI for `GuardianOfTruth.score(prompt, answer)`.
            Enter a prompt and a model answer, then inspect the predicted hallucination probability and timing breakdown.
            """
        )
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt", lines=5, placeholder="Enter the user question or prompt.")
            answer_box = gr.Textbox(label="Answer", lines=7, placeholder="Enter the model answer to score.")
        with gr.Row():
            submit = gr.Button("Score", variant="primary")
            gr.ClearButton([prompt_box, answer_box], value="Clear")

        verdict_box = gr.Textbox(label="Verdict", interactive=False)
        details_box = gr.JSON(label="Scoring result")

        submit.click(
            fn=lambda prompt, answer: run_score(prompt, answer, model_dir=model_dir),
            inputs=[prompt_box, answer_box],
            outputs=[verdict_box, details_box],
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[prompt_box, answer_box],
            outputs=[verdict_box, details_box],
            fn=lambda prompt, answer: run_score(prompt, answer, model_dir=model_dir),
            cache_examples=False,
        )
    return demo


def launch_demo(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    model_dir: str = "model",
    share: bool = True,
    inbrowser: bool = False,
) -> None:
    demo = build_demo(model_dir=model_dir)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Guardian of Truth Gradio frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--share", dest="share", action="store_true")
    parser.add_argument("--no-share", dest="share", action="store_false")
    parser.add_argument("--inbrowser", action="store_true")
    parser.set_defaults(share=True)
    args = parser.parse_args()

    launch_demo(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()
