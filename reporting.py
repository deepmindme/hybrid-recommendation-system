"""Графики и пояснения к фактическим таблицам эксперимента."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def md_table(table, digits=4):
    frame = table.copy()
    headers = [frame.index.name or 'Модель', *map(str, frame.columns)]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for index, row in frame.iterrows():
        values = []
        for v in row:
            if isinstance(v, (int, float, np.number)):
                values.append('—' if pd.isna(v) else f'{v:.{digits}f}')
            else: values.append(str(v))
        lines.append('| ' + ' | '.join([str(index), *values]) + ' |')
    return '\n'.join(lines)


def conclusions(tables, report):
    table = tables['model_comparison']
    final = table.loc['Hybrid + MLP + MMR']
    classic = table.loc['Hybrid']
    base_nn = table.loc['MLP scores only']
    full_nn = table.loc['MLP full']
    best = table['nDCG@5'].idxmax()
    recall_best = table['Recall@10'].idxmax()
    ratio = (final['ILD@10'] / table.loc['Hybrid + MLP', 'ILD@10'] - 1) * 100
    ci = report['feature_CI95']
    extra = 'интервал не включает ноль' if ci[0] > 0 else 'интервал включает ноль' if ci[1] >= 0 else 'интервал целиком ниже нуля'
    return f'''### Что получилось

1. По nDCG@5 лучше всего оказался **{best}**: **{table.loc[best, 'nDCG@5']:.4f}**. Классический гибрид получил {classic['nDCG@5']:.4f}, итоговый гибрид с MLP и MMR — {final['nDCG@5']:.4f}. Самая сложная система не обязательно выигрывает по каждой метрике.
2. Вариант **{recall_best}** лучше по Recall@10: **{table.loc[recall_best, 'Recall@10']:.4f}**. У итогового гибрида Precision@5 = {final['Precision@5']:.4f}, nDCG@10 = {final['nDCG@10']:.4f}. Поэтому выбор зависит от того, насколько важны первые позиции или более полный список.
3. Расширенные признаки помогли MLP на test: nDCG@5 вырос с **{base_nn['nDCG@5']:.4f}** до **{full_nn['nDCG@5']:.4f}**. Парная разность — {report['feature_difference']:.5f}, 95% bootstrap-интервал [{ci[0]:.5f}; {ci[1]:.5f}]: {extra}. Но на validation контрольный MLP был лучше. После переобучения базовых моделей признаки изменились, поэтому этот эффект стоит проверить на других разбиениях.
4. MMR увеличил тематическое разнообразие итоговой выдачи на **{ratio:.1f}%**: ILD@10 изменился с {table.loc['Hybrid + MLP', 'ILD@10']:.4f} до {final['ILD@10']:.4f}. На validation разрешалось потерять не больше 2% nDCG@5; это ограничение не гарантирует такую же границу на новых данных.
5. В среднем пул содержит {table.candidate_count.iloc[0]:.0f} книг и **{report['candidate_recall']:.1%}** релевантных тестовых книг пользователя. Оставшуюся часть ранжировщик вернуть не может. Увеличение полноты кандидатов — отдельное направление улучшения.
6. Для полностью новых пользователей все модели используют Popularity: ни одна модель не знает их вкусы. Для новых книг помогает контентная ветка. По отдельной таблице видно, что MLP без этой ветки практически не находит новые книги.
7. Item-CF для прогноза рейтинга и CF Retrieval решают разные задачи. Первый усредняет отклонения оценок, второй суммирует положительные связи по тем же соседям. Для top-N второй вариант оказался заметно полезнее. SVD, обученный на ошибке рейтинга, тоже не обязан хорошо ранжировать весь каталог.

### Ограничения и идеи

Дат оценок нет, а локальный файл ограничен примерно 100 оценками на книгу. Поэтому реальная популярность и будущая активность здесь отражены не полностью. Теги считаются доступными заранее, хотя исторических снимков нет.

Для обучения MLP я беру до десяти книг без положительной отложенной оценки на одну положительную. Это условные отрицательные примеры, а не доказательство, что книги пользователю не понравились. Полученную вероятность нельзя считать настоящей вероятностью интереса. Из 3500 выбранных пользователей у {report['sampling']['users_without_hits']} не нашлось положительной книги в пуле, поэтому они не дали обучающих строк MLP.

Выбор параметров сделан на {report['validation_users']} validation-пользователях, тест — на {report['ranking_users']}. Bootstrap учитывает разброс по этим пользователям, но не разные разбиения и повторное обучение. Разбиение повторяет ДЗ1 для сопоставимости: общие свойства этого теста уже были известны по первой работе.

В дальнейшем можно попробвать OOF-признаки для ранжировщика на нескольких фолдах, более сильную генерацию кандидатов и подбор весов отдельно по сегментам. Для контента — аннотации и Sentence-BERT. Затем нужны временное разбиение и проверка на реальных пользователях.
'''


def make_figures(tables, project_dir):
    output = Path(project_dir) / 'figures'; output.mkdir(exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    t = tables['model_comparison']
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    t[['Precision@5', 'Recall@5', 'nDCG@5']].plot.bar(ax=axes[0], rot=35)
    t[['Recall@10', 'nDCG@10']].plot.bar(ax=axes[1], rot=35)
    for ax, k in zip(axes, [5, 10]):
        ax.set_title(f'Сравнение рекомендаций, K={k}'); ax.set_ylabel('Значение метрики'); ax.set_xlabel('Модель')
    plt.tight_layout(); fig.savefig(output / '01_model_comparison.png', dpi=130); plt.close(fig)

    history = tables['neural_history']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model, part in history.groupby('model'):
        label = 'Только оценки моделей' if model == 'nn_base' else 'Все 34 признака'
        axes[0].plot(part.epoch, part.training_loss, 'o-', label=label)
        axes[1].plot(part.epoch, part.validation_nDCG5, 'o-', label=label)
    for ax in axes: ax.set_xlabel('Эпоха'); ax.legend()
    axes[0].set_title('Обучение MLP'); axes[0].set_ylabel('Binary cross-entropy')
    axes[1].set_title('Выбор эпохи на validation'); axes[1].set_ylabel('nDCG@5')
    plt.tight_layout(); fig.savefig(output / '02_neural_learning.png', dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in tables['diversity_validation'].groupby('model'):
        ax.plot(group.ILD10, group.nDCG5, 'o-', label=name)
        for _, row in group.iterrows(): ax.annotate(f"λ={row.mmr_weight:.1f}", (row.ILD10, row.nDCG5), xytext=(4, 5), textcoords='offset points')
    ax.set_xlabel('Тематическое разнообразие ILD@10'); ax.set_ylabel('nDCG@5')
    ax.set_title('Компромисс точности и разнообразия на validation'); ax.legend()
    plt.tight_layout(); fig.savefig(output / '03_diversity_tradeoff.png', dpi=130); plt.close(fig)

    names = ['Content-Based', 'CF Retrieval', 'MLP full', 'Hybrid + MLP + MMR']
    segments = tables['history_segments'].reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    segments[segments.model.isin(names)].pivot(index='segment', columns='model', values='nDCG5').plot.bar(ax=ax, rot=0)
    ax.set_xlabel('Число оценок в обучении'); ax.set_ylabel('nDCG@5'); ax.set_title('Разная длина истории пользователя')
    plt.tight_layout(); fig.savefig(output / '04_user_segments.png', dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scenario, title in zip(axes, ['new_users', 'new_books'], ['Новые пользователи', 'Новые книги']):
        tables['cold_start'].xs(scenario).loc[names, 'nDCG@5'].plot.bar(ax=ax, rot=25)
        ax.set_title(title); ax.set_ylabel('nDCG@5'); ax.set_xlabel('Модель')
    plt.tight_layout(); fig.savefig(output / '05_cold_start.png', dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    t.loc[['MLP scores only', 'MLP full'], ['nDCG@5', 'Recall@10']].plot.bar(ax=axes[0], rot=0)
    axes[0].set_title('Проверка расширенных признаков'); axes[0].set_ylabel('Значение метрики'); axes[0].set_xlabel('Вариант MLP')
    t.loc[['Hybrid + MLP', 'Hybrid + MLP + MMR'], ['ILD@10', 'Coverage@10']].plot.bar(ax=axes[1], rot=10)
    axes[1].set_title('Влияние переранжирования MMR'); axes[1].set_ylabel('Значение метрики'); axes[1].set_xlabel('Вариант гибрида')
    plt.tight_layout(); fig.savefig(output / '06_feature_and_diversity_ablation.png', dpi=130); plt.close(fig)
    return sorted(output.glob('*.png'))


def write_documents(tables, report, project_dir):
    root = Path(project_dir)
    text = conclusions(tables, report)
    (root / 'results' / 'conclusions.md').write_text(text, encoding='utf-8')
    comparison = tables['model_comparison']
    rows = ['Content-Based', 'CF Retrieval', 'MLP full', 'Hybrid + MLP + MMR']
    summary = comparison.loc[rows, ['nDCG@5', 'Recall@10', 'ILD@10']].copy()
    summary.index = ['По содержанию', 'CF Retrieval', 'Нейросеть', 'Итоговый гибрид']
    summary.index.name = 'Модель'
    readme = f'''# Рекомендации книг

Это продолжение первой работы на Goodbooks-10k. В ней я сравнивал отдельные модели, а здесь попробовал объединить их и добавить нейросеть. Хотел проверить, поможет ли это находить больше подходящих книг и меньше повторяться в рекомендациях.

Весь ход работы, графики и выводы есть в [ноутбуке](hybrid_recommendation_system.ipynb).

## Как работает решение

Сначала Popularity, TF-IDF, Item-CF, SVD и CF Retrieval предлагают книги. От каждого метода беру до 80 вариантов и объединяю их в один список. Уже оценённые пользователем книги убираю. CF Retrieval подбирает книги по связям с теми, которым пользователь поставил высокую оценку.

Затем считаю {report['feature_count']} признака: оценки этих моделей, статистики пользователя и книги, совпадение тем с историей чтения. На них обучаю небольшую нейросеть MLP с двумя скрытыми слоями. Её оценку смешиваю со взвешенной оценкой базовых моделей.

В конце применяю MMR: он немного меняет порядок, чтобы в списке было меньше похожих книг. Если у пользователя совсем нет истории, показываю популярные книги. Для новых книг использую их содержание.

Для обучения, подбора параметров и проверки использую разные части данных. Нейросеть обучаю на отдельно отложенной части train, которую базовые модели до этого не видели. Параметры выбираю на validation, итоговые метрики считаю на test.

## Что получилось

Проверка сделана на {report['ranking_users']} пользователях. Подходящей считаю книгу с оценкой 4 или 5. nDCG@5 показывает качество первых пяти рекомендаций с учётом порядка, Recall@10 — какую долю подходящих книг удалось найти в первой десятке, ILD@10 — насколько книги в ней отличаются по темам. Чем выше эти метрики, тем лучше.

{md_table(summary)}

По nDCG@5 лучше оказался CF Retrieval, а итоговый гибрид нашёл больше подходящих книг в первой десятке. MMR сделал список разнообразнее по сравнению с тем же гибридом без MMR. То есть добавление нейросети не дало выигрыша по всем показателям.

У данных есть ограничения: нет дат оценок, а в локальной версии примерно по 100 оценок на книгу. Поэтому проверить рекомендации на будущих событиях и точно оценить популярность здесь не получится. Полное сравнение моделей и остальные ограничения описаны в ноутбуке.

## Как запустить

Использовал Python 3.13.7. Данные уже лежат в `data/`. Из папки проекта выполнить:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name rec-system-2 --display-name "Python (rec-system-2)"
python -m notebook hybrid_recommendation_system.ipynb
```

В Jupyter выбрать ядро `Python (rec-system-2)` и запустить все ячейки по порядку. Результаты и графики в ноутбуке уже сохранены, их можно посмотреть без повторного обучения.

Можно запустить обучение и проверку из терминала:

```bash
python run_pipeline.py --config config.json
python audit.py
```

## Файлы

- `recommender.py` — подготовка данных, модели, рекомендации и метрики.
- `run_pipeline.py` и `config.json` — запуск эксперимента и его параметры.
- `reporting.py` — построение графиков и сохранение пояснений.
- `audit.py` — проверки результатов после обучения.
- `data/` — исходные CSV, `results/` — таблицы с результатами, `figures/` — графики.
'''
    (root / 'README.md').write_text(readme, encoding='utf-8')
    return text
