import os
from pathlib import Path

from guardian_of_truth.utils import load_local_env


def test_load_local_env_reads_dotenv_files(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "GROQ_API_KEY=test-key",
                "export APP_MODE=demo",
                "QUOTED_VALUE='hello world'",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("APP_MODE", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)

    updates = load_local_env(env_path)

    assert updates["GROQ_API_KEY"] == "test-key"
    assert os.environ["GROQ_API_KEY"] == "test-key"
    assert os.environ["APP_MODE"] == "demo"
    assert os.environ["QUOTED_VALUE"] == "hello world"


def test_load_local_env_does_not_override_existing_values(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("GROQ_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", "existing-key")

    updates = load_local_env(env_path)

    assert updates == {}
    assert os.environ["GROQ_API_KEY"] == "existing-key"
