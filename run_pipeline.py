"""Полный повтор эксперимента: python run_pipeline.py --config config.json."""
import argparse
import json
from pathlib import Path
from recommender import Config, run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    config_file = Path(args.config)
    if not config_file.is_absolute(): config_file = directory / config_file
    config = Config(**json.loads(config_file.read_text(encoding='utf-8')))
    _, tables, report = run_experiment(directory, config)
    print(tables['model_comparison'].round(4).to_string())
    print('Проверки:', report['checks'])


if __name__ == '__main__':
    main()
