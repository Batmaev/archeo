# Задача

Нужно найти 10 классов на изображениях и сохранить полигоны в geojson.

Классы:
1. Городища
2. Фортификации
3. Архитектура
4. Селища
5. Курганы
6. Дороги
7. Ямы
8. Пашни
9. Межа
10. Иное

Большая часть данных — geotiff лидарные снимки.

У некоторых регионов есть спутниковая и аэрофотосъемка. На первом этапе можно ее игнорировать.


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

5. Рассчитать метрики:
```bash
# Создать ground truth файл
python3 metrics/merge_ground_truth.py --input-root train/002_ДЕМИДОВКА_FINAL --output metrics/ground_truth.geojson

# Вычислить метрику для квалификационного этапа
python3 metrics/compute_metrics_qual.py --predictions baseline_solution/result.geojson --ground-truth metrics/ground_truth.geojson

# Вычислить метрику для финала
python3 metrics/compute_metrics.py --predictions baseline_solution/result.geojson --ground-truth metrics/ground_truth.geojson
```