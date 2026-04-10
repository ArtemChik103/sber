# Guardian of Truth

Текущий статус проекта на момент последнего обновления:

- `pytest -q` проходит: `23 passed, 1 skipped`
- лучший подтвержденный результат на полном public bench: `PR-AUC = 0.5617`
- для private bench уже сгенерирован и запушен [knowledge_bench_private_scores.csv](knowledge_bench_private_scores.csv)
- есть one-click launcher с простым Gradio UI и публичной ссылкой по умолчанию

API-only детектор фактологических галлюцинаций с baseline-совместимым интерфейсом `GuardianOfTruth.score(prompt, answer)`. Проект использует Groq как внешний verifier, компактные API/text features и локальный lightweight classifier поверх них.

## Что Уже Готово

На текущий момент в проекте реализовано:

- runtime API `GuardianOfTruth.score(prompt, answer) -> ScoringResult`
- Groq verifier с кешем, rate limiting и fallback path
- feature pipeline для API-сигналов и локальных text features
- training pipeline на synthetic JSONL без обучения на public bench
- ingestion внешних factual datasets `PopQA` и `FEVER`
- generation pipeline для `seed`, `rule_negative`, `groq_negative` и targeted augmentation
- sequential public/private scoring с checkpointing
- простой Gradio demo frontend
- one-click launcher на `python` и `.bat`
- unit/integration tests и CI workflow

## Структура Репозитория

- [src/guardian_of_truth](src/guardian_of_truth) - основная логика verifier, features, classifier, runtime scoring и evaluation
- [configs](configs) - конфиги API, feature set и training/model settings
- [data](data) - synthetic/raw data, public bench и SQLite cache
- [model](model) - локальные model artifacts и training summaries
- [tests](tests) - unit, integration и UI smoke tests
- [scripts](scripts) - shell automation для install, train, scoring и dataset generation
- [run_project.py](run_project.py), [run_project.bat](run_project.bat) - one-click запуск frontend
- [train.py](train.py), [evaluate.py](evaluate.py), [app.py](app.py) - корневые entrypoints

## Архитектура

Поток runtime scoring выглядит так:

1. Клиент вызывает `GuardianOfTruth.score(prompt, answer)`.
2. `GroqVerifier` делает короткий audit-вызов к `llama-3.1-8b-instant`.
3. Verifier возвращает компактный JSON audit с hallucination/relevance/contradiction/question-fit сигналами.
4. `FeatureExtractor` собирает API-features и prompt-aware text features.
5. Локальный classifier выдает `predict_proba`.
6. Если API path неуспешен, включается text-only fallback classifier.
7. Runtime возвращает `ScoringResult` с вероятностью и таймингами.

Ключевые точки в коде:

- runtime scoring: [guardian.py](src/guardian_of_truth/guardian.py)
- Groq client и audit schema: [api_client.py](src/guardian_of_truth/api_client.py)
- feature extraction: [feature_extractor.py](src/guardian_of_truth/feature_extractor.py)
- preprocessing / matrix build: [preprocess.py](src/guardian_of_truth/preprocess.py)
- classifier и calibration: [classifier.py](src/guardian_of_truth/classifier.py)
- training helpers: [training.py](src/guardian_of_truth/training.py)
- synthetic generation: [generation.py](src/guardian_of_truth/generation.py)
- external dataset ingestion: [external_data.py](src/guardian_of_truth/external_data.py)
- evaluation / public scoring: [evaluate.py](src/guardian_of_truth/evaluate.py)
- Gradio UI: [gradio_app.py](src/guardian_of_truth/gradio_app.py)

## Публичный Контракт

Основной runtime-контракт:

```python
GuardianOfTruth.score(prompt: str, answer: str) -> ScoringResult
```

Возвращаемая структура:

```python
ScoringResult(
    is_hallucination: bool,
    is_hallucination_proba: float,
    t_model_sec: float,
    t_overhead_sec: float,
    t_total_sec: float,
)
```

Вспомогательные интерфейсы:

- `GroqVerifier.verify(prompt, answer, mode)`
- `FeatureExtractor.extract(prompt, answer, audit)`

Стабильные runtime-инварианты:

- `predict_proba` всегда в диапазоне `[0, 1]`
- `fallback` не должен падать при `timeout`, `429`, invalid JSON или отсутствии API key
- public bench не используется в `fit`
- private/public scoring сохраняет исходные строки и добавляет `predict_proba`

## Качество И Текущие Чекпоинты

Главные факты по quality:

- лучший исторически подтвержденный full-public результат: `PR-AUC = 0.5617`
- текущий runtime-код использует совместимый локальный чекпоинт в `model/`
- private bench скоринг уже пересчитан и сохранен в [knowledge_bench_private_scores.csv](knowledge_bench_private_scores.csv)

Важное уточнение:

- у проекта есть старые и новые feature spaces
- лучший исторический public чекпоинт и текущий runtime-чекпоинт не обязаны быть одним и тем же физическим набором артефактов
- private scoring уже проверялся и на текущем runtime-чекпоинте, и на старом `best_public_v2`; итоговые `predict_proba` совпали по всем `1038` строкам

## Данные

Основные источники данных:

- [data/raw/seed_qa.jsonl](data/raw/seed_qa.jsonl) - seed factual QA
- [data/raw/synthetic_factual_data.jsonl](data/raw/synthetic_factual_data.jsonl) - synthetic training corpus
- [data/bench/knowledge_bench_public.csv](data/bench/knowledge_bench_public.csv) - public bench только для evaluation
- [data/cache/groq_cache.sqlite](data/cache/groq_cache.sqlite) - Groq cache

Поддерживаемые variant types в synthetic data:

- `positive`
- `rule_negative`
- `groq_negative`
- `popqa_positive`
- `popqa_negative`
- `fever_supports`
- `fever_refutes`
- `groq_supported_positive`
- `groq_drift_negative`

Стабильное правило по данным:

- [data/bench/knowledge_bench_public.csv](data/bench/knowledge_bench_public.csv) не используется для train/calibration

## Быстрый Запуск

Если окружение уже подготовлено, самый короткий запуск:

```powershell
python run_project.py
```

На Windows можно так:

```bat
run_project.bat
```

Что делает launcher:

- запускает простой Gradio UI
- по умолчанию включает `share=True`
- печатает локальную и публичную ссылку

Если публичная ссылка не нужна:

```powershell
python run_project.py --no-share
```

## Подготовка Чистой Машины

Минимально нужно:

- `Git`
- `Python 3.11`
- доступ в интернет для Groq API и Gradio share link

### Проверка Базовых Инструментов

В PowerShell:

```powershell
git --version
python --version
```

### Первичная Установка Зависимостей

После `git clone`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Если нужен реальный API path, перед запуском задай:

```powershell
$env:GROQ_API_KEY="your_key_here"
```

## Варианты Запуска

### Вариант 1. Один Командный Запуск

- Python: `python run_project.py`
- Windows batch: `run_project.bat`

Полезные флаги:

- `--host 127.0.0.1`
- `--port 7860`
- `--no-share`
- `--inbrowser`

### Вариант 2. Прямой Запуск UI

```powershell
python app.py
python -m guardian_of_truth.gradio_app
```

### Вариант 3. Shell Scripts

```bash
./scripts/run_ui.sh
./scripts/smoke.sh
./scripts/train.sh
./scripts/score_public.sh --csv-path data/bench/knowledge_bench_public.csv
```

На Windows PowerShell:

```powershell
.\scripts\run_ui.ps1
```

## Обучение И Скоринг

Train на synthetic data:

```powershell
python train.py --dataset-path data/raw/synthetic_factual_data.jsonl
```

Ограниченный train для быстрых экспериментов:

```powershell
python train.py --dataset-path data/raw/synthetic_factual_data.jsonl --limit 300
```

Public scoring:

```powershell
python evaluate.py --csv-path data/bench/knowledge_bench_public.csv --output-path outputs/public_scored.csv
```

Быстрый deterministic dev-slice:

```powershell
python -m guardian_of_truth.evaluate --dev-slice-size 150 --slice-name balanced --output-path outputs/public_dev150_balanced.csv
python -m guardian_of_truth.evaluate --dev-slice-size 150 --slice-name typed --output-path outputs/public_dev150_typed.csv
```

Private scoring уже сгенерирован:

- [knowledge_bench_private_scores.csv](knowledge_bench_private_scores.csv)

## Генерация Датасета

External ingestion:

```bash
./scripts/ingest_external.sh --stage all --popqa-limit 500 --fever-limit 500 --resume --merge-main
```

Synthetic generation:

```bash
./scripts/generate_dataset.sh --stage seed-harvest --resume --limit 300
./scripts/generate_dataset.sh --stage rule-negatives --resume
./scripts/generate_dataset.sh --stage groq-negatives --resume --limit 100
```

Pipeline time estimate:

```bash
./scripts/estimate_pipeline.sh --planned-groq-negatives 300
```

## Автоматические Проверки

### Полный Локальный Прогон

```powershell
pytest -q
```

Текущий статус:

- `23 passed, 1 skipped`

Что покрыто:

- API client normalization
- feature extraction
- preprocess / stratified sampling
- dataset generation helpers
- training helpers
- evaluate/dev-slice logic
- runtime guardian fallback behavior
- Gradio UI smoke
- import smoke
- optional live Groq integration test при наличии `GROQ_API_KEY`

Ключевые test files:

- [test_api_client.py](tests/test_api_client.py)
- [test_feature_extractor.py](tests/test_feature_extractor.py)
- [test_preprocess.py](tests/test_preprocess.py)
- [test_generation.py](tests/test_generation.py)
- [test_training.py](tests/test_training.py)
- [test_evaluate.py](tests/test_evaluate.py)
- [test_guardian.py](tests/test_guardian.py)
- [test_gradio_app.py](tests/test_gradio_app.py)
- [test_import.py](tests/test_import.py)
- [test_integration_live.py](tests/test_integration_live.py)

### CI

Workflow:

- [.github/workflows/ci.yml](.github/workflows/ci.yml)

Что делает CI:

- ставит зависимости
- делает import smoke
- запускает `pytest -q`

## Как Проверять Проект Экспертам

Самый короткий маршрут проверки:

1. Установить зависимости через `pip install -r requirements.txt`
2. Задать `GROQ_API_KEY`
3. Выполнить `pytest -q`
4. Выполнить `python run_project.py`
5. Открыть локальную или публичную Gradio-ссылку
6. Проверить пару factual / hallucination кейсов вручную
7. При необходимости прогнать `evaluate.py` на public bench

## Операционные Замечания

- проект API-only и не использует локальную LLM-инференсную модель
- public bench зарезервирован под evaluation, а не под train
- `data/cache/groq_cache.sqlite` нужен для повторного использования API-audits
- full public scoring на free-plan Groq может занимать десятки минут
- честный live latency зависит от сети и внешнего API; без жёсткого cutoff SLA `<500ms` не гарантируется
- приватный входной bench файл намеренно не добавляется в git
