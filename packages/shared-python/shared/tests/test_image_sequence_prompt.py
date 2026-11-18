from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("TMP_PATH", "/tmp/knowhere-test")
os.environ.setdefault("S3_BUCKET_NAME", "test-uploads")
os.environ.setdefault("S3_ACCESS_KEY_ID", "test")
os.environ.setdefault("S3_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("S3_TEMP_PATH", "/tmp")

from shared.services.ai.prompt_service import build_prompt


def test_summary_images_prompt_stays_single_image_without_sequence() -> None:
    prompt, *_ = build_prompt(
        "summary-images",
        "",
        "",
        paras={"max_tokens": 200},
    )
    transcribe, *_ = build_prompt("transcribe", "", "", paras={"max_tokens": 100})
    assert "single image" in prompt
    assert "frames in temporal or page order" not in prompt
    assert "verbatim body text" in transcribe
    assert "frames in temporal or page order" not in transcribe


def test_summary_images_prompt_describes_gif_sequence() -> None:
    prompt, *_ = build_prompt(
        "summary-images",
        "",
        "",
        paras={
            "max_tokens": 200,
            "sequence_kind": "gif",
            "sequence_frames": 8,
        },
    )
    assert "8 frames" in prompt
    assert "animated GIF" in prompt
    assert "single image" not in prompt


def test_judge_and_transcribe_prompts_describe_sequence() -> None:
    judge, *_ = build_prompt(
        "judge-image-type",
        "",
        "",
        paras={"sequence_kind": "tiff_pages", "sequence_frames": 4},
    )
    transcribe, *_ = build_prompt(
        "transcribe",
        "",
        "",
        paras={"max_tokens": 100, "sequence_kind": "animated_webp", "sequence_frames": 6},
    )
    assert "4 frames" in judge
    assert "multi-page TIFF" in judge
    assert "6 frames" in transcribe
    assert "animated WebP" in transcribe


def test_formula_redelimit_prompt_builds_with_braced_latex_examples() -> None:
    """The prompt embeds literal {1/2} / {U} braces; .format must survive."""
    prompt, temperature, _top_p, max_tokens = build_prompt(
        "formula-redelimit",
        "A(t) = [U(t)^2 + V(t)^2]^(1/2)",
        "",
    )
    assert "formula-redelimit" not in prompt
    assert "A(t) = [U(t)^2 + V(t)^2]^(1/2)" in prompt
    assert "x^{1/2}" in prompt
    assert "\dot{U}(t)" in prompt
    assert temperature == 0.1
    assert max_tokens == 2000
