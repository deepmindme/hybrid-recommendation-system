"""Модели из первого ДЗ и общий пул кандидатов для итогового проекта."""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from surprise import Dataset, Reader, SVD


@dataclass
class Config:
    seed: int = 42
    cold_users: int = 300
    cold_books: int = 100
    rank_train_users: int = 3500
    valid_users: int = 600
    test_users: int = 2500
    cold_eval_users: int = 1000
    per_source: int = 80
    neighbors: int = 50
    min_ratings: int = 30
    factors: int = 64
    svd_epochs: int = 30
    svd_reg: float = 0.1
    mlp_epochs: int = 12
    negative_ratio: int = 10
    diversity_budget: float = 0.02


GENRES = ['fantasy', 'romance', 'mystery', 'science-fiction', 'historical-fiction',
          'young-adult', 'classics', 'thriller', 'horror', 'non-fiction',
          'biography', 'history', 'poetry', 'children', 'comics', 'philosophy']
BASE_NAMES = ['Popularity', 'Content-Based', 'Item-Based CF', 'SVD', 'CF Retrieval']
FEATURE_NAMES = ['pop_mean', 'content_cosine', 'cf_prediction', 'svd_prediction', 'cf_positive',
                 'user_mean', 'user_log_count', 'user_positive_fraction', 'user_std', 'user_topic_entropy',
                 'book_log_count', 'book_mean', 'book_std', 'book_rating_entropy', 'book_positive_fraction',
                 'topic_affinity', 'mean_rating_gap', 'cf_support', *['genre_' + x for x in GENRES]]


def stable_top(scores, k):
    order = np.argsort(-np.asarray(scores), kind='stable')
    return order[np.isfinite(np.asarray(scores)[order])][:k]


def precision(recs, relevant, k):
    return sum(x in relevant for x in recs[:k]) / k


def recall(recs, relevant, k):
    return sum(x in relevant for x in recs[:k]) / len(relevant) if relevant else 0.0


def ndcg(recs, relevant, k):
    if not relevant:
        return 0.0
    hits = np.array([x in relevant for x in recs[:k]], dtype=float)
    dcg = (hits / np.log2(np.arange(len(hits)) + 2)).sum()
    ideal = (1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum()
    return float(dcg / ideal)


def split_users(frame, seed):
    shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    counts = shuffled.groupby('user_id').book_id.transform('size')
    n = np.where(counts >= 5, np.maximum(1, (counts * 0.2).astype(int)), 0)
    held = shuffled.groupby('user_id').cumcount() < n
    return shuffled[~held].reset_index(drop=True), shuffled[held].reset_index(drop=True)


def entropy_rows(counts):
    probabilities = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)


class BookData:
    def __init__(self, data_dir):
        directory = Path(data_dir)
        books = pd.read_csv(directory / 'books.csv').sort_values('id').reset_index(drop=True)
        ratings = pd.read_csv(directory / 'ratings.csv').drop_duplicates(['user_id', 'book_id'], keep='last')
        tags = pd.read_csv(directory / 'tags.csv')
        bt = pd.read_csv(directory / 'book_tags.csv').drop_duplicates().merge(tags, on='tag_id', validate='many_to_one')
        assert books.id.is_unique and books.book_id.is_unique
        assert not ratings.isna().any().any() and ratings.rating.between(1, 5).all()
        assert set(ratings.book_id).issubset(set(books.id))
        # Перевожу исходные ID в последовательные индексы: так можно менять CSV.
        self.user_ids = np.sort(ratings.user_id.unique())
        self.user_map = {int(x): i + 1 for i, x in enumerate(self.user_ids)}
        self.book_map = {int(x): i + 1 for i, x in enumerate(books.id)}
        ratings = ratings.copy()
        ratings.user_id = ratings.user_id.map(self.user_map)
        ratings.book_id = ratings.book_id.map(self.book_map)
        self.ratings = ratings.reset_index(drop=True)
        self.n_users, self.n_books = len(self.user_ids), len(books)
        books['clean_title'] = books.original_title.fillna(books.title).fillna('')
        stop = {'to-read', 'currently-reading', 'favorites', 'favourites', 'books-i-own', 'owned',
                'owned-books', 'library', 'my-books', 'to-buy', 'default', 'kindle', 'ebook', 'ebooks',
                'audiobook', 'audiobooks', 'audio', 'wish-list', 'my-library', 'i-own', 'own-it', 'read', 'have'}
        bt.tag_name = bt.tag_name.fillna('').str.lower().str.strip()
        clean = bt[~bt.tag_name.isin(stop) & (bt.tag_name.str.len() > 1)]
        top = (clean[clean['count'] >= 5].sort_values(['goodreads_book_id', 'count'], ascending=[True, False])
               .groupby('goodreads_book_id').head(30))
        text = top.groupby('goodreads_book_id').tag_name.apply(lambda s: ' '.join(s.str.replace('-', ' ', regex=False)))
        profiles = books.clean_title + ' ' + books.authors.fillna('') + ' ' + books.book_id.map(text).fillna('')
        self.vectorizer = TfidfVectorizer(max_features=15000, min_df=2, max_df=.85, ngram_range=(1, 2),
                                         stop_words='english', sublinear_tf=True, dtype=np.float32)
        self.tfidf = self.vectorizer.fit_transform(profiles).tocsr()
        genre_table = (clean[clean.tag_name.isin(GENRES)].pivot_table(index='goodreads_book_id', columns='tag_name',
                       values='count', aggfunc='sum', fill_value=0).reindex(columns=GENRES, fill_value=0))
        genre_values = genre_table.reindex(books.book_id, fill_value=0).to_numpy(dtype=np.float32)
        self.genres = normalize(np.log1p(genre_values), norm='l2').astype(np.float32)
        self.authors = pd.factorize(books.authors.fillna('').str.split(',').str[0])[0]
        self.books = books

    def split(self, cfg):
        rng = np.random.default_rng(cfg.seed)
        counts = self.ratings.groupby('user_id').size()
        candidates = counts[counts >= 5].index.to_numpy()
        cold_u = rng.choice(candidates, min(cfg.cold_users, max(1, len(candidates) // 10)), replace=False)
        cold_i = rng.choice(np.arange(1, self.n_books + 1), min(cfg.cold_books, max(1, self.n_books // 10)), replace=False)
        cu = self.ratings.user_id.isin(cold_u)
        ci = self.ratings.book_id.isin(cold_i)
        main = self.ratings[~cu & ~ci]
        train_valid, test = split_users(main, cfg.seed)
        train, valid = split_users(train_valid, cfg.seed + 1)
        core, rank_train = split_users(train, cfg.seed + 10)
        parts = dict(core=core, rank_train=rank_train, validation=valid, test=test,
                     cold_users=self.ratings[cu].copy(), cold_books=self.ratings[~cu & ci].copy())
        indexes = {name: pd.MultiIndex.from_frame(df[['user_id', 'book_id']]) for name, df in parts.items()}
        for a in parts:
            for b in parts:
                if a < b:
                    assert indexes[a].intersection(indexes[b]).empty
        assert sum(map(len, parts.values())) == len(self.ratings)
        self.cold_user_ids, self.cold_book_ids = np.sort(cold_u), np.sort(cold_i)
        return parts, train_valid


class Engine:
    """Четыре модели ДЗ1 и признаки, вычисленные только по текущему обучению."""
    def __init__(self, data, frame, cfg):
        self.data, self.cfg = data, cfg
        self.n_users, self.n_books = data.n_users, data.n_books
        u = frame.user_id.to_numpy() - 1
        i = frame.book_id.to_numpy() - 1
        r = frame.rating.to_numpy(dtype=np.float32)
        self.R = sparse.csr_matrix((r, (u, i)), shape=(self.n_users, self.n_books))
        self.seen = self.R.copy(); self.seen.data[:] = 1
        self.positive = self.R.copy()
        self.positive.data = np.where(self.positive.data >= 4, self.positive.data - 3, 0)
        self.positive.eliminate_zeros()
        self.uc = np.asarray(self.seen.sum(axis=1)).ravel()
        self.ic = np.asarray(self.seen.sum(axis=0)).ravel()
        mean = float(r.mean())
        self.um = np.divide(np.asarray(self.R.sum(axis=1)).ravel(), self.uc,
                            out=np.full(self.n_users, mean, dtype=np.float32), where=self.uc > 0)
        self.im = np.divide(np.asarray(self.R.sum(axis=0)).ravel(), self.ic,
                            out=np.full(self.n_books, mean, dtype=np.float32), where=self.ic > 0)
        self.centered = sparse.csr_matrix((r - self.um[u], (u, i)), shape=self.R.shape)
        squared = self.R.copy(); squared.data **= 2
        user_std = np.sqrt(np.maximum(np.asarray(squared.sum(axis=1)).ravel() / np.maximum(self.uc, 1) - self.um ** 2, 0))
        book_std = np.sqrt(np.maximum(np.asarray(squared.sum(axis=0)).ravel() / np.maximum(self.ic, 1) - self.im ** 2, 0))
        self.profiles = normalize(self.positive @ data.tfidf, norm='l2').tocsr()
        self.user_genres = normalize(np.asarray(self.positive @ data.genres), norm='l2')
        user_likes = self.positive.getnnz(axis=1) / np.maximum(self.uc, 1)
        book_likes = self.positive.getnnz(axis=0) / np.maximum(self.ic, 1)
        rating_hist = np.column_stack([np.bincount(i[r == value], minlength=self.n_books) for value in range(1, 6)])
        self.uf = np.column_stack([self.um, np.log1p(self.uc), user_likes, user_std, entropy_rows(self.user_genres)]).astype(np.float32)
        self.bf = np.column_stack([np.log1p(self.ic), self.im, book_std, entropy_rows(rating_hist), book_likes]).astype(np.float32)
        self.pop_allowed = self.ic >= cfg.min_ratings
        if not self.pop_allowed.any(): self.pop_allowed = self.ic > 0
        self.S = self._neighbors()
        trainset = Dataset.load_from_df(frame[['user_id', 'book_id', 'rating']], Reader(rating_scale=(1, 5))).build_full_trainset()
        algo = SVD(n_factors=cfg.factors, n_epochs=cfg.svd_epochs, reg_all=cfg.svd_reg,
                   lr_all=.007, random_state=cfg.seed).fit(trainset)
        ru = np.array([int(trainset.to_raw_uid(x)) - 1 for x in trainset.all_users()])
        ri = np.array([int(trainset.to_raw_iid(x)) - 1 for x in trainset.all_items()])
        self.mu = float(trainset.global_mean)
        self.bu = np.zeros(self.n_users); self.bi = np.zeros(self.n_books)
        self.P = np.zeros((self.n_users, cfg.factors)); self.Q = np.zeros((self.n_books, cfg.factors))
        self.bu[ru], self.bi[ri], self.P[ru], self.Q[ri] = algo.bu, algo.bi, algo.pu, algo.qi
        for a, b in zip(u[:5], i[:5]):
            prediction = np.clip(self.mu + self.bu[a] + self.bi[b] + self.P[a] @ self.Q[b], 1, 5)
            assert np.isclose(prediction, algo.predict(int(a + 1), int(b + 1)).est)

    def _neighbors(self):
        vectors = normalize(self.R.T.tocsr(), norm='l2')
        rows, cols, values = [], [], []
        k = min(self.cfg.neighbors, self.n_books - 1)
        for start in range(0, self.n_books, 256):
            end = min(start + 256, self.n_books)
            sim = (vectors[start:end] @ vectors.T).toarray()
            sim[np.arange(end - start), np.arange(start, end)] = 0
            idx = np.argpartition(sim, -k, axis=1)[:, -k:]
            rr = np.repeat(np.arange(start, end), k); cc = idx.ravel()
            vv = sim[np.repeat(np.arange(end - start), k), cc]
            keep = vv > 0
            rows.append(rr[keep]); cols.append(cc[keep]); values.append(vv[keep])
        return sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(self.n_books, self.n_books), dtype=np.float32)

    def raw_scores(self, users, allow_cold=False):
        u = np.asarray(users, dtype=int) - 1
        pop = np.broadcast_to(self.im, (len(u), self.n_books)).copy()
        cb = (self.profiles[u] @ self.data.tfidf.T).toarray()
        no_likes = self.positive[u].getnnz(axis=1) == 0
        cb[no_likes] = pop[no_likes]
        numerator = (self.centered[u] @ self.S.T).toarray()
        support = (self.seen[u] @ self.S.T).toarray()
        fallback = .5 * self.um[u, None] + .5 * self.im[None, :]
        cf = np.clip(np.where(support > 1e-6, self.um[u, None] + numerator / np.maximum(support, 1e-6), fallback), 1, 5)
        svd = np.clip(self.mu + self.bu[u, None] + self.bi[None, :] + self.P[u] @ self.Q.T, 1, 5)
        retrieval = (self.positive[u] @ self.S.T).toarray()
        values = np.stack([pop, cb, cf, svd, retrieval], axis=2).astype(np.float32)
        available = ~self.seen[u].toarray().astype(bool)
        if not allow_cold: available &= (self.ic > 0)[None, :]
        cold_user = self.uc[u] == 0
        values[cold_user] = pop[cold_user, :, None]
        return values, available, support

    def candidate_cache(self, frame, users, allow_cold=False):
        relevant = frame[frame.rating >= 4].groupby('user_id').book_id.apply(lambda s: set(s.to_numpy() - 1)).to_dict()
        rows = []
        for start in range(0, len(users), 32):
            batch = users[start:start + 32]
            raw, available, support = self.raw_scores(batch, allow_cold)
            for b, user in enumerate(batch):
                u = int(user) - 1
                pools = []
                for source in range(5):
                    scores = raw[b, :, source].copy()
                    allowed = available[b].copy()
                    if source == 0 or self.uc[u] == 0 or (source == 1 and self.positive[u].nnz == 0):
                        allowed &= self.pop_allowed
                    scores[~allowed] = -np.inf
                    pools.append(stable_top(scores, self.cfg.per_source))
                ids = np.unique(np.concatenate(pools))
                base = raw[b, ids]
                topic = self.data.genres[ids] @ self.user_genres[u]
                extra = np.column_stack([topic, np.abs(self.um[u] - self.im[ids]), support[b, ids]])
                X = np.column_stack([base, np.tile(self.uf[u], (len(ids), 1)), self.bf[ids], extra, self.data.genres[ids]]).astype(np.float32)
                assert X.shape[1] == len(FEATURE_NAMES) and np.isfinite(X).all()
                assert not set(ids) & set(self.seen[u].indices)
                rows.append(dict(user_id=int(user), ids=ids, X=X, relevant=relevant.get(int(user), set()),
                                 count=float(self.uc[u]), no_likes=self.positive[u].nnz == 0))
        return rows


def choose_users(frame, engine, limit, seed, require_history=True):
    positive = frame[frame.rating >= 4].user_id.unique()
    users = np.sort([int(u) for u in positive if not require_history or engine.uc[int(u) - 1] > 0])
    return np.sort(np.random.default_rng(seed).choice(users, min(limit, len(users)), replace=False))


def scale_row(scores):
    return (scores - scores.min()) / max(float(np.ptp(scores)), 1e-8)


def classic_score(row, engine, weights):
    if row['count'] == 0:
        return row['X'][:, 0].copy()
    effective = np.array([.20, .65, 0, 0, .15]) if row['count'] < 5 else np.array(weights)
    normalized = np.column_stack([scale_row(row['X'][:, j]) for j in range(5)])
    score = normalized @ effective
    cold = engine.ic[row['ids']] == 0
    score[cold] = normalized[cold, 1]
    return score


def mmr(ids, scores, data, weight=1.0, k=10):
    short = stable_top(scores, min(100, len(ids)))
    if weight == 1 or len(short) == 0:
        return ids[short[:k]]
    relevance = scale_row(scores[short])
    redundancy = np.zeros(len(short))
    chosen, used = [], np.zeros(len(short), dtype=bool)
    for _ in range(min(k, len(short))):
        value = weight * relevance - (1 - weight) * redundancy
        value[used] = -np.inf
        pick = int(np.argmax(value)); used[pick] = True
        chosen.append(ids[short[pick]])
        topic_similarity = data.genres[ids[short]] @ data.genres[ids[short[pick]]]
        same_author = (data.authors[ids[short]] == data.authors[ids[short[pick]]]).astype(float)
        redundancy = np.maximum(redundancy, np.maximum(topic_similarity, .8 * same_author))
    return np.array(chosen)


def model_recommendations(row, engine, name, weights, alpha=0, mmr_weight=1.0):
    if row['count'] == 0:
        scores = row['X'][:, 0].copy()
        scores[~engine.pop_allowed[row['ids']]] = -np.inf
        return row['ids'][stable_top(scores, 10)]
    if name in BASE_NAMES:
        scores = row['X'][:, BASE_NAMES.index(name)].copy()
        if name == 'Popularity' or (name == 'Content-Based' and row['no_likes']):
            scores[~engine.pop_allowed[row['ids']]] = -np.inf
        return row['ids'][stable_top(scores, 10)]
    if name == 'MLP scores only':
        return row['ids'][stable_top(row['nn_base'], 10)]
    if name == 'MLP full':
        return row['ids'][stable_top(row['nn'], 10)]
    score = classic_score(row, engine, weights)
    if name in ('Hybrid + MLP', 'Hybrid + MLP + MMR'):
        score = (1 - alpha) * scale_row(score) + alpha * scale_row(row['nn'])
        cold = engine.ic[row['ids']] == 0
        score[cold] = scale_row(row['X'][:, 1])[cold]
    diversity = mmr_weight if name.endswith('+ MMR') else 1.0
    return mmr(row['ids'], score, engine.data, diversity)


def add_neural_scores(cache, rankers):
    if not cache: return
    X = np.concatenate([row['X'] for row in cache])
    for key, (scaler, model, columns) in rankers.items():
        predictions = model.predict_proba(scaler.transform(X[:, columns]))[:, 1]
        offset = 0
        for row in cache:
            row[key] = predictions[offset:offset + len(row['ids'])]
            offset += len(row['ids'])


def training_pairs(cache, cfg):
    rng = np.random.default_rng(cfg.seed + 20)
    features, labels = [], []
    hit_users, missed_users = 0, 0
    for row in cache:
        positive = np.flatnonzero(np.isin(row['ids'], list(row['relevant'])))
        if len(positive) == 0:
            missed_users += 1
            continue
        negative = np.flatnonzero(~np.isin(row['ids'], list(row['relevant'])))
        negative = rng.choice(negative, min(len(negative), cfg.negative_ratio * len(positive)), replace=False)
        chosen = np.concatenate([positive, negative])
        features.append(row['X'][chosen])
        labels.append(np.concatenate([np.ones(len(positive)), np.zeros(len(negative))]))
        hit_users += 1
    if not features: raise ValueError('В обучающем пуле нет положительных примеров')
    X, y = np.concatenate(features), np.concatenate(labels).astype(int)
    assert set(y) == {0, 1}
    return X, y, dict(users_with_hits=hit_users, users_without_hits=missed_users,
                     examples=len(y), positives=int(y.sum()), negatives=int((y == 0).sum()))


def fit_rankers(train_cache, valid_cache, cfg):
    X, y, sampling = training_pairs(train_cache, cfg)
    val_X = np.concatenate([r['X'] for r in valid_cache])
    rankers, history = {}, []
    for key, columns in [('nn_base', np.arange(5)), ('nn', np.arange(len(FEATURE_NAMES)))]:
        scaler = StandardScaler().fit(X[:, columns])
        xt, xv = scaler.transform(X[:, columns]), scaler.transform(val_X[:, columns])
        model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                              batch_size=512, learning_rate_init=.001, alpha=.01,
                              early_stopping=False, random_state=cfg.seed)
        best, best_value, best_epoch, stale = None, -np.inf, 0, 0
        for epoch in range(1, cfg.mlp_epochs + 1):
            model.partial_fit(xt, y, classes=[0, 1])
            predictions = model.predict_proba(xv)[:, 1]
            values, offset = [], 0
            for row in valid_cache:
                part = predictions[offset:offset + len(row['ids'])]; offset += len(row['ids'])
                recs = row['ids'][stable_top(part, 10)]
                values.append(ndcg(recs, row['relevant'], 5))
            score = float(np.mean(values))
            history.append(dict(model=key, epoch=epoch, training_loss=float(model.loss_), validation_nDCG5=score))
            if score > best_value + 1e-6:
                best, best_value, best_epoch, stale = copy.deepcopy(model), score, epoch, 0
            else:
                stale += 1
            if stale >= 3: break
        rankers[key] = (scaler, best, columns)
        sampling[key + '_epoch'] = best_epoch
        sampling[key + '_validation_nDCG5'] = best_value
    return rankers, pd.DataFrame(history), sampling


def diversity(recs, data):
    if len(recs) < 2: return 0.0
    g = data.genres[recs]
    valid = np.linalg.norm(g, axis=1) > 0
    pair = np.triu(np.outer(valid, valid), 1)
    if not pair.any(): return np.nan
    return float(1 - (g @ g.T)[pair].mean())


def evaluate(cache, engine, methods, weights, alpha=0, mmr_weight=1.0):
    records, coverage = [], {name: set() for name in methods}
    for row in cache:
        for name in methods:
            recs = model_recommendations(row, engine, name, weights, alpha, mmr_weight)
            assert len(recs) == len(set(recs)) and set(recs).issubset(set(row['ids']))
            relevant = row['relevant']
            value = dict(model=name, user_id=row['user_id'], history_count=row['count'],
                         candidate_count=len(row['ids']), CandidateRecall=len(set(row['ids']) & relevant) / len(relevant) if relevant else 0)
            for k in (5, 10):
                value[f'Precision@{k}'] = precision(recs, relevant, k)
                value[f'Recall@{k}'] = recall(recs, relevant, k)
                value[f'nDCG@{k}'] = ndcg(recs, relevant, k)
            value['ILD@10'] = diversity(recs, engine.data)
            value['Authors@10'] = len(set(engine.data.authors[recs]))
            coverage[name].update(recs)
            records.append(value)
    per_user = pd.DataFrame(records)
    table = per_user.groupby('model', sort=False).mean(numeric_only=True).drop(columns=['user_id', 'history_count'])
    table['Coverage@10'] = [len(coverage[name]) / engine.n_books for name in table.index]
    table['users'] = len(cache)
    return table, per_user


def optimize(valid_cache, engine, cfg):
    weight_grid = [(0.05, .30, .05, .05, .55), (.05, .50, .05, .05, .35),
                   (.05, .20, .05, .20, .50), (.10, .35, .05, .10, .40)]
    classical = []
    for weights in weight_grid:
        table, _ = evaluate(valid_cache, engine, ['Hybrid'], weights)
        classical.append(dict(weights=list(weights), nDCG5=table.loc['Hybrid', 'nDCG@5']))
    weights = tuple(max(classical, key=lambda x: x['nDCG5'])['weights'])
    neural = []
    for alpha in (0., .25, .5, .75, 1.):
        table, _ = evaluate(valid_cache, engine, ['Hybrid + MLP'], weights, alpha)
        neural.append(dict(alpha=alpha, nDCG5=table.loc['Hybrid + MLP', 'nDCG@5']))
    alpha = max(neural, key=lambda x: x['nDCG5'])['alpha']
    tradeoffs = []
    for name in ['Hybrid + MMR', 'Hybrid + MLP + MMR']:
        for weight in (.8, .9, 1.):
            table, _ = evaluate(valid_cache, engine, [name], weights, alpha, weight)
            tradeoffs.append(dict(model=name, mmr_weight=weight, nDCG5=table.loc[name, 'nDCG@5'],
                                 ILD10=table.loc[name, 'ILD@10'], Coverage10=table.loc[name, 'Coverage@10']))
    tradeoffs = pd.DataFrame(tradeoffs)
    selected = {}
    for name in ['Hybrid + MMR', 'Hybrid + MLP + MMR']:
        subset = tradeoffs[tradeoffs.model == name]
        reference = subset.loc[subset.mmr_weight == 1., 'nDCG5'].iloc[0]
        eligible = subset[subset.nDCG5 >= (1 - cfg.diversity_budget) * reference]
        selected[name] = float(eligible.sort_values(['ILD10', 'nDCG5'], ascending=False).iloc[0].mmr_weight)
    return dict(weights=weights, alpha=float(alpha), mmr=selected), pd.DataFrame(classical), pd.DataFrame(neural), tradeoffs


def evaluate_selected(cache, engine, selected):
    unbalanced = [*BASE_NAMES, 'MLP scores only', 'MLP full', 'Hybrid', 'Hybrid + MLP']
    table, per_user = evaluate(cache, engine, unbalanced, selected['weights'], selected['alpha'])
    for name, mmr_weight in selected['mmr'].items():
        t, p = evaluate(cache, engine, [name], selected['weights'], selected['alpha'], mmr_weight)
        table = pd.concat([table, t]); per_user = pd.concat([per_user, p], ignore_index=True)
    return table, per_user


class Recommender:
    def __init__(self, engine, rankers, selected):
        self.engine, self.rankers, self.selected = engine, rankers, selected

    def recommend(self, user_id, N=5, model='Hybrid + MLP + MMR', allow_new_books=True):
        if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or not 1 <= N <= 10:
            raise ValueError('N должно быть целым от 1 до 10')
        if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)) or user_id < 1:
            raise ValueError('user_id должен быть положительным целым')
        methods = [*BASE_NAMES, 'MLP scores only', 'MLP full', 'Hybrid', 'Hybrid + MMR', 'Hybrid + MLP', 'Hybrid + MLP + MMR']
        if model not in methods: raise ValueError('Неизвестная модель')
        internal = self.engine.data.user_map.get(int(user_id))
        if internal is None:
            ids = stable_top(np.where(self.engine.pop_allowed, self.engine.im, -np.inf), N)
        else:
            empty = pd.DataFrame(columns=['user_id', 'book_id', 'rating'])
            cache = self.engine.candidate_cache(empty, [internal], allow_new_books)
            add_neural_scores(cache, self.rankers)
            row = cache[0]
            ids = model_recommendations(row, self.engine, model, self.selected['weights'], self.selected['alpha'],
                                        self.selected['mmr'].get(model, 1.))[:N]
        result = self.engine.data.books.iloc[ids][['id', 'clean_title', 'authors']].copy()
        return result.reset_index(drop=True)


def validate_result(bundle, data, parts, test_cache, cfg):
    assert precision([1, 2, 3, 4, 5], {1, 3, 8}, 5) == .4
    assert recall([1, 2, 3], {1, 3, 8}, 3) == 2/3
    assert np.isclose(ndcg([1, 2, 3, 4, 5], {1, 3, 8}, 5), 1.5 / (1 + 1 / np.log2(3) + .5))
    assert ndcg([1], set(), 5) == 0
    assert np.isclose(ndcg([1, 2], {1, 2}, 2), 1)
    engine = bundle.engine
    assert (engine.uc[data.cold_user_ids - 1] == 0).all()
    assert (engine.ic[data.cold_book_ids - 1] == 0).all()
    for row in test_cache[:8]:
        public_id = int(data.user_ids[row['user_id'] - 1])
        for name in [*BASE_NAMES, 'Hybrid', 'MLP full', 'Hybrid + MLP + MMR']:
            actual = bundle.recommend(public_id, 10, name, allow_new_books=False)
            idx = model_recommendations(row, engine, name, bundle.selected['weights'], bundle.selected['alpha'],
                                        bundle.selected['mmr'].get(name, 1.))
            assert np.array_equal(actual.id.to_numpy(), data.books.iloc[idx].id.to_numpy()), (public_id, name)
            assert not set(idx) & set(engine.seen[row['user_id'] - 1].indices)
    unknown = int(max(data.user_ids)) + 1000
    assert bundle.recommend(unknown).id.tolist() == bundle.recommend(unknown, model='Popularity').id.tolist()
    for invalid in [0, -1, True, 1.5]:
        try:
            bundle.recommend(invalid)
            raise AssertionError('Принят неправильный ID')
        except ValueError: pass
    return dict(split_disjoint=True, metric_examples=True, no_seen_items=True,
                public_api_matches_evaluation=True, cold_holdouts=True, unknown_user_fallback=True)


def run_experiment(project_dir='.', cfg=None):
    cfg = cfg or Config()
    directory = Path(project_dir)
    result_dir = directory / 'results'; result_dir.mkdir(exist_ok=True)
    start = time.perf_counter()
    with threadpool_limits(limits=4):
        print('1/7: данные и разбиение', flush=True)
        data = BookData(directory / 'data')
        parts, final_train = data.split(cfg)
        print('2/7: базовые модели на внутреннем train', flush=True)
        selection_engine = Engine(data, parts['core'], cfg)
        rank_users = choose_users(parts['rank_train'], selection_engine, cfg.rank_train_users, cfg.seed + 11)
        valid_users = choose_users(parts['validation'], selection_engine, cfg.valid_users, cfg.seed + 12)
        train_cache = selection_engine.candidate_cache(parts['rank_train'], rank_users)
        valid_cache = selection_engine.candidate_cache(parts['validation'], valid_users)
        # Ни одна целевая оценка ранжировщика не входит в историю его базовых моделей.
        for row in train_cache:
            assert not row['relevant'] & set(selection_engine.seen[row['user_id'] - 1].indices)
        print('3/7: MLP с расширенными признаками и контрольный вариант', flush=True)
        rankers, history, sampling = fit_rankers(train_cache, valid_cache, cfg)
        add_neural_scores(valid_cache, rankers)
        print('4/7: подбор весов и разнообразия на validation', flush=True)
        selected, weights_grid, alpha_grid, diversity_grid = optimize(valid_cache, selection_engine, cfg)
        valid_table, _ = evaluate_selected(valid_cache, selection_engine, selected)
        # Сохраняю примеры признаков, а не все внутренние обучающие строки.
        features_preview = pd.DataFrame(train_cache[0]['X'][:20], columns=FEATURE_NAMES)
        feature_stats = pd.DataFrame(np.concatenate([r['X'] for r in train_cache]), columns=FEATURE_NAMES).describe().T
        del train_cache, valid_cache, selection_engine
        print('5/7: финальные базовые модели и тест', flush=True)
        final_engine = Engine(data, final_train, cfg)
        test_users = choose_users(parts['test'], final_engine, cfg.test_users, cfg.seed + 3)
        test_cache = final_engine.candidate_cache(parts['test'], test_users)
        add_neural_scores(test_cache, rankers)
        table, per_user = evaluate_selected(test_cache, final_engine, selected)
        bundle = Recommender(final_engine, rankers, selected)
        print('6/7: новые пользователи и книги', flush=True)
        cu_users = choose_users(parts['cold_users'], final_engine, cfg.cold_users, cfg.seed + 4, False)
        ci_users = choose_users(parts['cold_books'], final_engine, cfg.cold_eval_users, cfg.seed + 5)
        cold_tables = []
        for name, users, frame in [('new_users', cu_users, parts['cold_users']), ('new_books', ci_users, parts['cold_books'])]:
            cache = final_engine.candidate_cache(frame, users, True)
            add_neural_scores(cache, rankers)
            t, _ = evaluate_selected(cache, final_engine, selected)
            t['scenario'] = name; cold_tables.append(t.reset_index())
        cold_table = pd.concat(cold_tables, ignore_index=True).set_index(['scenario', 'model'])
        print('7/7: проверки и сохранение результатов', flush=True)
        checks = validate_result(bundle, data, parts, test_cache, cfg)
        groups = pd.cut(per_user.history_count, [0, 4, 14, np.inf], labels=['1–4', '5–14', '15+'])
        segments = per_user.assign(segment=groups).groupby(['segment', 'model'], observed=True).agg(
            users=('user_id', 'size'), nDCG5=('nDCG@5', 'mean'), Precision5=('Precision@5', 'mean'),
            Recall5=('Recall@5', 'mean'), ILD10=('ILD@10', 'mean'))
        selected_user_table = pd.DataFrame({'internal_user_id': test_users, 'source_user_id': data.user_ids[test_users - 1]})
        best_validation_single = valid_table.loc[[*BASE_NAMES, 'MLP full'], 'nDCG@5'].idxmax()
        paired = per_user.pivot(index='user_id', columns='model', values='nDCG@5')
        delta = (paired['Hybrid + MLP + MMR'] - paired[best_validation_single]).to_numpy()
        rng = np.random.default_rng(cfg.seed + 30)
        draws = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
        interval = np.quantile(draws, [.025, .975])
        nn_delta = (paired['MLP full'] - paired['MLP scores only']).to_numpy()
        ablation_draws = np.array([rng.choice(nn_delta, len(nn_delta), replace=True).mean() for _ in range(2000)])
        feature_interval = np.quantile(ablation_draws, [.025, .975])
        outputs = dict(model_comparison=table, validation_comparison=valid_table, per_user_metrics=per_user,
                       history_segments=segments, cold_start=cold_table, neural_history=history,
                       hybrid_weights_validation=weights_grid, neural_weight_validation=alpha_grid,
                       diversity_validation=diversity_grid, feature_examples=features_preview,
                       feature_statistics=feature_stats, evaluated_users=selected_user_table)
        for name, frame in outputs.items(): frame.to_csv(result_dir / (name + '.csv'))
        feature_dictionary = pd.DataFrame({'feature': FEATURE_NAMES, 'group': ['base_score'] * 5 + ['user'] * 5 + ['book'] * 5 + ['interaction'] * 3 + ['topic'] * len(GENRES)})
        feature_dictionary.to_csv(result_dir / 'feature_dictionary.csv', index=False)
        report = dict(config=asdict(cfg), splits={k: len(v) for k, v in parts.items()},
                      final_training_ratings=len(final_train), dataset_ratings=len(data.ratings),
                      users=data.n_users, books=data.n_books, selected=selected, sampling=sampling,
                      ranking_users=len(test_users), validation_users=len(valid_users), feature_count=len(FEATURE_NAMES),
                      candidate_recall=float(table.CandidateRecall.iloc[0]), checks=checks,
                      comparison_model=best_validation_single, hybrid_difference=float(delta.mean()),
                      hybrid_CI95=interval.tolist(), feature_difference=float(nn_delta.mean()),
                      feature_CI95=feature_interval.tolist(), elapsed_seconds=time.perf_counter() - start)
        (result_dir / 'run_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        import joblib
        (directory / 'artifacts').mkdir(exist_ok=True)
        joblib.dump(bundle, directory / 'artifacts' / 'recommender.joblib', compress=3)
        return bundle, outputs, report
