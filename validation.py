"""
validation.py — валидация ранжирований и выбор числа признаков.

Содержит класс ModelValidator: кривая качества модели на топ-N признаков
ранжирования (или упорядоченного набора) для каждого N из ns, выбор
оптимального N и итогового набора.

Принципы:
    * Всегда фиксированный self.cv (без cv_for_run): кривые по разным
      ранжированиям и моделям сравниваются на одинаковых фолдах, поэтому
      оценки парные (нужно для Friedman/Nemenyi в statistics.py).
    * Seed модели на фолде = fold_index — одинаковый для всех N и всех
      label'ов, чтобы разница между точками кривых не включала шум seed.
    * Параллелится только цикл по (N, fold) — n_jobs. Многопоточные модели
      (XGBoost, LightGBM) нужно делать однопоточными через n_jobs=1
      в kwargs фабрики.
    * Метрики — строки sklearn-scorer'ов (get_scorer): у всех scorer'ов
      «больше — лучше» (neg_log_loss и т.п.), поэтому максимум ищется
      одинаково для любой метрики.

Пример использования:
    X, y, feature_names, groups, classes = load_data("data.csv", "Y")
    ev = ImportanceEvaluator(X, y, feature_names).compute_mi("mi")
    val = ModelValidator(X, y, feature_names, random_state=42)
    val.evaluate_model_on_subsets(
        lambda s: make_rf(s, n_estimators=100),
        ev.get_ranking("mi"), "val_rf_mi",
        scoring=["f1_macro", "roc_auc_ovr"], ns=[1, 3, 5, 7, 10],
    )
    subset = val.get_best_subset("val_rf_mi", "f1_macro")
    n = val.get_optimal_n_features("val_rf_mi", "f1_macro")  # само число
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import get_scorer

from common import (
    ConfigurationError,
    check_cv,
    check_data,
    make_folds,
)

# Настройка формата/уровня — на стороне пользователя, как в common.py.
logger = logging.getLogger("otbor.validation")
logger.addHandler(logging.NullHandler())


class ModelValidator:
    """Кривые качества модели по числу признаков на общих фолдах.

    Каждый evaluate_model_on_subsets сохраняет результат под своим label:
        validation_results[label] — dict:
            "features": list[str] — упорядоченный список признаков;
            "ns": list[int] — оценённые N (по возрастанию);
            "scores": {metric: np.ndarray (len(ns), n_folds)} — метрика
                модели на топ-N признаков в каждом фолде.

    Параметры
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
        Коды классов 0..K-1 (см. load_data).
    feature_names : list[str]
        Имена признаков, len == n_features.
    groups : array-like, shape (n_samples,) | None, default None
        Группы для групповых сплиттеров (GroupKFold, LeaveOneGroupOut).
    cv : sklearn CV splitter | None, default None
        Проходит через check_cv(cv, random_state); None ->
        StratifiedKFold(5, shuffle=True, random_state=random_state).
    random_state : int, default 42
        Seed дефолтного сплиттера (при cv=None).
    n_jobs : int, default -1
        Число параллельных задач (N, fold); -1 — все ядра.

    Исключения
    ----------
    DataValidationError
        Несогласованные размеры X, y, groups, feature_names.
    ConfigurationError
        Некорректный cv.
    """

    def __init__(self, X, y, feature_names, groups=None, cv=None,
                 random_state=42, n_jobs=-1):
        X, y, feature_names, groups = check_data(X, y, feature_names, groups)

        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.groups = groups
        self.cv = check_cv(cv, random_state)
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.validation_results = {}

    # -----------------------------------------------------------------------
    # Каркас
    # -----------------------------------------------------------------------

    @staticmethod
    def _run_fold(model_factory, X, y, columns, train_indices, val_indices,
                  seed, scoring):
        """Одна задача (N, fold): обучение на топ-N признаков и метрики на val.

        Выполняется в процессе-воркере joblib: всё нужное передаётся
        аргументами, результат — только через return.

        Возвращает
        -------
        dict
            {metric: score}.
        """
        X_cols = X[:, columns]
        model = model_factory(seed)
        model.fit(X_cols[train_indices], y[train_indices])
        X_val, y_val = X_cols[val_indices], y[val_indices]
        return {metric: get_scorer(metric)(model, X_val, y_val)
                for metric in scoring}

    def _get_result(self, label):
        """validation_results[label] с понятной ошибкой при отсутствии."""
        if label not in self.validation_results:
            raise ConfigurationError(
                f"Нет результатов для label={label!r}. "
                f"Доступны: {list(self.validation_results)}"
            )
        return self.validation_results[label]

    # -----------------------------------------------------------------------
    # Методы
    # -----------------------------------------------------------------------

    def evaluate_model_on_subsets(self, model_factory, ranking_or_subset,
                                  label, scoring=("f1_macro",), ns=None,
                                  check_val_classes=True):
        """Качество модели на топ-N признаков для каждого N из ns.

        Список признаков считается упорядоченным: топ-N — первые N его
        элементов. Подходит ранжирование (get_ranking(...)["feature"]) и
        упорядоченный результат перебора (forward, RFE). Для
        неупорядоченного набора передаётся ns=[len(subset)].

        Параметры
        ----------
        model_factory : callable
            seed -> модель, например lambda s: make_rf(s, n_estimators=100).
        ranking_or_subset : list[str] | pd.DataFrame
            Имена признаков из feature_names, без повторов; DataFrame
            (результат get_ranking) — берётся его колонка 'feature'.
        label : str
            Уникальный label результата.
        scoring : str | list[str], default ('f1_macro',)
            Строки sklearn-scorer'ов; строка заворачивается в список.
        ns : list[int] | None, default None
            Числа признаков, 1 <= N <= len(ranking_or_subset); повторы
            удаляются, порядок — по возрастанию. None -> все N от 1 до
            len(ranking_or_subset).
        check_val_classes : bool, default True
            Требовать все классы в каждом val-фолде (нужно для roc_auc_ovr
            и neg_log_loss, без этого f1_macro по фолдам несопоставим).
            При LOGO, где val — одна группа (скважина), можно отключить.

        Возвращает
        -------
        self

        Исключения
        ----------
        ConfigurationError
            label уже использован; N вне диапазона.
        """
        if label in self.validation_results:
            raise ConfigurationError(f"label={label!r} уже использован")

        scoring = [scoring] if isinstance(scoring, str) else list(scoring)

        # DataFrame из get_ranking: list(df) дал бы имена колонок.
        if isinstance(ranking_or_subset, pd.DataFrame):
            ranking_or_subset = ranking_or_subset["feature"]
        features = list(ranking_or_subset)

        if ns is None:
            ns = range(1, len(features) + 1)
        ns = sorted(set(ns))
        bad = [n for n in ns if not 1 <= n <= len(features)]
        if bad:
            raise ConfigurationError(
                f"ns: значения {bad} вне диапазона 1..{len(features)}"
            )

        # Индексы колонок X в порядке ранжирования: топ-N = order[:N].
        order = [self.feature_names.index(f) for f in features]
        folds = make_folds(self.cv, self.X, self.y, self.groups,
                           check_val_classes=check_val_classes)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_fold)(model_factory, self.X, self.y, order[:n],
                                    train_indices, val_indices,
                                    fold_index, scoring)
            for n in ns
            for fold_index, (train_indices, val_indices) in enumerate(folds)
        )

        # results идут в порядке (N, fold) -> матрица (len(ns), n_folds).
        scores = {
            metric: np.array([r[metric] for r in results],
                             dtype=np.float64).reshape(len(ns), len(folds))
            for metric in scoring
        }
        self.validation_results[label] = {
            "features": features,
            "ns": ns,
            "scores": scores,
        }
        logger.info(f"{label}: готово")
        return self

    def get_optimal_n_features(self, label, metric, rule="1se"):
        """Оптимальное число признаков по кривой качества.

        Правила (mean, SE — среднее и стандартная ошибка метрики по
        фолдам, SE = std(ddof=1) / sqrt(n_folds)):
            'max' — N с наибольшим mean; при равенстве — меньшее N;
            '1se' — наименьшее N, у которого mean >= mean* - SE*, где
                    mean*, SE* — у N с наибольшим mean. Точки в пределах
                    одной SE от максимума неотличимы на фоне шума CV;
                    из них выбирается самый компактный набор.

        Параметры
        ----------
        label : str
        metric : str
            Одна из метрик, переданных в evaluate_model_on_subsets.
        rule : {'1se', 'max'}, default '1se'

        Возвращает
        -------
        n : int
        """
        if rule not in ("1se", "max"):
            raise ConfigurationError(
                f"rule должен быть '1se' или 'max', получено {rule!r}"
            )
        result = self._get_result(label)
        if metric not in result["scores"]:
            raise ConfigurationError(
                f"{label}: нет метрики {metric!r}. "
                f"Доступны: {list(result['scores'])}"
            )
        scores = result["scores"][metric]
        mean = scores.mean(axis=1)
        best = int(np.argmax(mean))   # argmax — первый максимум, т.е. меньшее N
        if rule == "max":
            return result["ns"][best]

        se = scores[best].std(ddof=1) / np.sqrt(scores.shape[1])
        within = np.flatnonzero(mean >= mean[best] - se)
        return result["ns"][int(within[0])]

    def get_best_subset(self, label, metric=None, rule="1se",
                        n_features=None):
        """Лучший набор: первые N признаков списка, оценённого под label.

        N находится по кривой качества (get_optimal_n_features(label,
        metric, rule)) или задаётся вручную через n_features — например,
        выбранное на глаз по get_validation_summary.

        Параметры
        ----------
        label : str
        metric : str | None, default None
            Метрика для выбора N; обязательна, если n_features не задан.
        rule : {'1se', 'max'}, default '1se'
            Правило выбора N (см. get_optimal_n_features).
        n_features : int | None, default None
            Заданное вручную N, 1 <= N <= длины списка; тогда metric и
            rule не используются.

        Возвращает
        -------
        subset : list[str]
        """
        features = self._get_result(label)["features"]
        if n_features is None:
            if metric is None:
                raise ConfigurationError(
                    "get_best_subset: задай metric (для выбора N) "
                    "или n_features"
                )
            n_features = self.get_optimal_n_features(label, metric, rule)
        if not 1 <= n_features <= len(features):
            raise ConfigurationError(
                f"n_features={n_features} вне диапазона 1..{len(features)}"
            )
        return features[:n_features]

    def get_validation_summary(self, label):
        """Сводка кривой качества: mean и std метрик по фолдам для каждого N.

        Параметры
        ----------
        label : str

        Возвращает
        -------
        summary : pd.DataFrame
            Колонки n_features, {metric}_mean, {metric}_std для каждой
            метрики; std по фолдам (ddof=1).
        """
        result = self._get_result(label)
        summary = pd.DataFrame({"n_features": result["ns"]})
        for metric, scores in result["scores"].items():
            summary[f"{metric}_mean"] = scores.mean(axis=1)
            summary[f"{metric}_std"] = scores.std(axis=1, ddof=1)
        return summary
