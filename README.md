# Guardian of Truth

Детектор фактологических галлюцинаций в генерациях языковых моделей с контрактом `GuardianOfTruth.score(prompt, answer)`.

Демо-приложение в облаке: [https://cyeux8q3oyqrzdksudrbss.streamlit.app/](https://cyeux8q3oyqrzdksudrbss.streamlit.app/)

## Метрики и статус валидации

- Набор тестов: `125 passed` (100% прохождение).
- Метрики на полном бенчмарке (1044 примера, `knowledge_bench_public.csv`):
  - **PR-AUC**: `0.9262` (прирост `+34.68 п.п.` к baseline `0.5794`).
  - **ROC-AUC**: `0.9187` (прирост `+32.47 п.п.` к baseline `0.5940`).
  - **Средняя задержка**: `189.2 мс` (требование SLA: `< 500 мс`).
- Приватный сабмишн: файл `knowledge_bench_private_scores.csv` сформирован и валидирован (1038 строк, диапазон скоров `[0.050, 0.950]`, пропуски отсутствуют).

### Сравнение с базовым решением

| Пайплайн | PR-AUC | ROC-AUC | Среднее время ответа | Примечание |
| :--- | :---: | :---: | :---: | :--- |
| Базовый классификатор | 0.5794 | 0.5940 | 15.2 мс | Baseline TF-IDF + Heuristics |
| Guardian of Truth v6 | 0.7652 | 0.7793 | 8.5 мс | Qwen 27B Direct + LogisticRegression |
| Guardian of Truth v8 | 0.8785 | 0.8696 | 181.0 мс | Qwen 27B Reasoning + Ансамбль деревьев |
| **Guardian of Truth v9.4** | **0.9262** | **0.9187** | **189.2 мс** | **Bayesian Blind Rescue + Морфологический стемминг + Изотоническая калибровка** |

## Архитектура системы

Система реализует многоуровневую верификацию ответов:

1. **Reasoning-верификатор (Groq API, модель `qwen/qwen3.8-27b`)**:
   - Формирует пошаговую цепочку проверки именованных сущностей, числовых значений и временных интервалов.
   - Использует балансировщик `MultiKeyRateLimiter` для распределения нагрузки между несколькими ключами API.
   - Сохраняет результаты в SQLite-кеш (`data/cache/groq_cache.sqlite`), исключая повторные сетевые вызовы.
2. **Экстрактор признаков (`FeatureExtractor`)**:
   - Извлекает 63 численных признака:
     - 16 структурированных признаков верификатора;
     - 27 признаков лексического анализа, перекрытия сущностей и маркеров неопределенности;
     - 20 признаков соответствия локальной базе знаний.
3. **Калиброванный ансамбль моделей (`model/`)**:
   - Объединяет `ExtraTreesClassifier`, `RandomForestClassifier` и `LogisticRegression`.
   - Калибрует итоговые вероятности через `CalibratedClassifierCV` и изотоническую регрессию (`isotonic_calibrator.joblib`).
4. **Байесовский rescue-контур (`Bayesian Blind Rescue Policy`)**:
   - Выполняет нормализацию падежей и окончаний русского языка через легковесный стеммер.
   - Снижает вероятность ошибки при совпадении ключевых фактов кандидата с независимой слепой генерацией.
5. **Автономный оффлайн-фоллбек (`fallback.joblib`)**:
   - Переключается на локальную классификацию при сетевых сбоях, таймаутах или отсутствии ключа API.

## Структура репозитория

```text
├── .streamlit/              # Конфигурация темы и параметров Streamlit
├── configs/                 # Конфигурационные файлы API и модели
├── data/
│   ├── bench/               # Датасеты knowledge_bench_public.csv
│   └── cache/               # База SQLite с кешем запросов Groq API
├── model/                   # Артефакты обученных классификаторов и калибраторов
├── outputs/                 # Таблицы скоринга публичного бенчмарка
├── src/guardian_of_truth/   # Исходный код пакета
│   ├── api_client.py        # Клиент Groq API с пулом ключей и обработкой лимитов
│   ├── classifier.py        # Архитектура классификатора и ансамбля
│   ├── feature_extractor.py # Вычисление 63 признаков текста и аудита
│   ├── guardian.py          # Основной класс GuardianOfTruth
│   ├── refute_overlay.py    # Байесовский rescue-контур и калибровка
│   └── streamlit_app.py     # Логика и компоненты веб-интерфейса
├── tests/                   # Набор автоматических тестов
├── evaluate.py              # Скрипт расчета метрик PR-AUC и ROC-AUC
├── pyproject.toml           # Метаданные пакета и зависимости
├── requirements.txt         # Список внешних библиотек
├── run_project.py           # Консольный лаунчер приложения
├── streamlit_app.py         # Точка входа для Streamlit Community Cloud
└── knowledge_bench_private_scores.csv # Итоговый файл сабмишна
```

## Программный интерфейс

Основной метод инференса:

```python
from guardian_of_truth import GuardianOfTruth

guardian = GuardianOfTruth()
result = guardian.score(
    prompt="В каком году был основан Санкт-Петербург?",
    answer="Санкт-Петербург был основан Петром I в 1703 году.",
)

print("Вердикт галлюцинации:", result.is_hallucination)
print("Вероятность галлюцинации:", result.is_hallucination_proba)
print("Время выполнения (сек):", result.t_total_sec)
```

Сигнатура структуры `ScoringResult`:

```python
ScoringResult(
    is_hallucination: bool,        # Бинарный вердикт (True при пороге >= 0.50)
    is_hallucination_proba: float, # Калиброванная вероятность галлюцинации [0, 1]
    t_model_sec: float,            # Время выполнения запроса к верификатору
    t_overhead_sec: float,         # Время извлечения признаков и инференса модели
    t_total_sec: float,            # Суммарное время обработки запроса
)
```

## Установка и запуск

### 1. Установка зависимостей

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Настройка переменных окружения

Создайте файл `.env` в корне репозитория:

```env
GROQ_API_KEY=gsk_your_groq_api_key
# Для пула ключей используйте перечисление через запятую:
# GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3
```

При работе с уже накопленным кешем запросов (`data/cache/groq_cache.sqlite`) сетевые обращения к API не производятся.

### 3. Запуск веб-интерфейса Streamlit

Запустите сервер интерфейса локально:

```bash
streamlit run streamlit_app.py
```

Или через скрипт-лаунчер:

```bash
python run_project.py
```

Приложение доступно по адресу `http://localhost:8501`.

### 4. Развертывание на Streamlit Community Cloud

1. Загрузите репозиторий на GitHub.
2. Подключите репозиторий в панели управления [share.streamlit.io](https://share.streamlit.io/).
3. Укажите параметры развертывания:
   - **Main file path**: `streamlit_app.py`
   - **Python version**: `3.11`
4. В разделе **Settings -> Secrets** добавьте ключ API:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key"
   ```
5. Нажмите **Deploy**. Приложение развертывается за 1–2 минуты, потребляет до 200 МБ оперативной памяти и укладывается в бесплатный тариф Streamlit Cloud.

### 5. Авто-пробуждение приложения (Keep-Alive)

Для предотвращения засыпания в бесплатном тарифе Streamlit Cloud настроен GitHub Action `.github/workflows/keep_alive.yml`:

- **Расписание**: запускается каждые 2 дня в 06:00 UTC.
- **Автономность**: целевой URL `https://cyeux8q3oyqrzdksudrbss.streamlit.app/` уже встроен по умолчанию и работает без добавления секретов или переменных.
- **Принцип работы**: открывает веб-страницу через headless Chromium (Playwright), определяет статус сна и автоматически нажимает кнопку пробуждения при её наличии.
- **Ручной запуск**: при необходимости можно запустить проверку вручную во вкладке **Actions -> keep-alive -> Run workflow**.

### 6. Воспроизведение метрик и запуск тестов

Расчет метрик v9.4 с изотонической калибровкой и стеммингом (PR-AUC 0.9262):

```bash
python scripts/evaluate_v9_public.py
```

Расчет метрик базового контура:

```bash
python evaluate.py --csv-path data/bench/knowledge_bench_public.csv --cache-only-api
```

Запуск автоматических тестов:

```bash
python -m pytest -q
```
