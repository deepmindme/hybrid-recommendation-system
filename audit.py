"""Проверки сохранённой системы без повторного обучения."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from recommender import BASE_NAMES, Config, BookData, model_recommendations, stable_top, mmr


def main():
    root = Path(__file__).resolve().parent
    bundle = joblib.load(root / 'artifacts' / 'recommender.joblib')
    engine = bundle.engine
    report = json.loads((root / 'results' / 'run_summary.json').read_text())
    ids = pd.read_csv(root / 'results' / 'evaluated_users.csv').internal_user_id.to_numpy()[:12]
    raw, available, _ = engine.raw_scores(ids)
    empty = pd.DataFrame(columns=['user_id', 'book_id', 'rating'])
    cache = engine.candidate_cache(empty, ids)
    for b, row in enumerate(cache):
        for source, name in enumerate(BASE_NAMES):
            score = raw[b, :, source].copy()
            allowed = available[b].copy()
            if name == 'Popularity' or (name == 'Content-Based' and row['no_likes']):
                allowed &= engine.pop_allowed
            score[~allowed] = -np.inf
            expected = stable_top(score, 10)
            actual = model_recommendations(row, engine, name, bundle.selected['weights'])
            assert np.array_equal(actual, expected), name
    # MMR с λ=1 должен сохранять сортировку релевантности.
    sample = cache[0]
    values = sample['X'][:, 1]
    assert np.array_equal(mmr(sample['ids'], values, engine.data, 1.), sample['ids'][stable_top(values, 10)])
    assert len(sample['ids']) == len(set(sample['ids']))
    assert all(report['checks'].values())
    table = pd.read_csv(root / 'results' / 'model_comparison.csv', index_col=0)
    assert table['users'].nunique() == 1 and table['users'].iloc[0] == report['ranking_users']
    assert table.CandidateRecall.nunique() == 1
    assert table['nDCG@5'].between(0, 1).all()
    assert table['Recall@10'].between(0, 1).all()
    assert table['Precision@5'].between(0, 1).all()
    assert np.all(table['Recall@10'] <= table.CandidateRecall + 1e-12)
    history = pd.read_csv(root / 'results' / 'neural_history.csv')
    for _, group in history.groupby('model'):
        assert group.training_loss.iloc[-1] < group.training_loss.iloc[0]
    for scaler, model, columns in bundle.rankers.values():
        assert scaler.n_features_in_ == len(columns)
        assert [x.shape for x in model.coefs_] == [(len(columns), 64), (64, 32), (32, 1)]
    cold = pd.read_csv(root / 'results' / 'cold_start.csv')
    new_users = cold[cold.scenario == 'new_users']
    assert new_users['nDCG@5'].nunique() == 1
    selected = report['selected']
    assert np.isclose(sum(selected['weights']), 1) and 0 <= selected['alpha'] <= 1
    tradeoff = pd.read_csv(root / 'results' / 'diversity_validation.csv')
    for name, weight in selected['mmr'].items():
        group = tradeoff[tradeoff.model == name]
        before = group.loc[group.mmr_weight == 1., 'nDCG5'].iloc[0]
        after = group.loc[group.mmr_weight == weight, 'nDCG5'].iloc[0]
        assert after >= (1 - report['config']['diversity_budget']) * before
    checks = dict(full_catalog_baselines_match_pool=True, mmr_identity=True, same_test_users=True,
                  candidate_recall_bound=True, neural_loss_decreased=True, mlp_architecture=True,
                  new_user_fallback=True, diversity_validation_constraint=True)
    (root / 'results' / 'audit.json').write_text(json.dumps(checks, indent=2) + '\n')
    print('Независимые проверки:', checks)


if __name__ == '__main__':
    main()
