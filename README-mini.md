# Запустить тренировку без скачивания 200 ГБ данных

1. Установить uv (менеджер пакетов для Python) — https://docs.astral.sh/uv/getting-started/installation/

2. Установить зависимости (и виртуальное окружение):
```bash
uv sync
```

3. Тестовая тренировка:
```bash
uv run baseline_recreation/train.py --epochs 2 --batch 8 --imgsz 768 --fraction 0.1 --ext-val-every 1
```

4. Полноценная тренировка:
```bash
uv run baseline_recreation/train.py --batch 128
```