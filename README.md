# Как запустить

1. Установить `uv` (мененджер пакетов для Python) — https://docs.astral.sh/uv/getting-started/installation/

2. Установить зависимости (и виртуальное окружение):
```bash
uv sync
```

3. Выгрузить данные с S3:
```bash
python3 downloading/download_s3_data.py --folder train --local-dir .
```

4. Запустить решение:
```bash
python3 baseline_solution/solution.py train/002_ДЕМИДОВКА_FINAL baseline_solution
```