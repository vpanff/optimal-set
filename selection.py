"""
selection.py — методы перебора признаков (Forward, Backward, RFE, SFS).

Содержит класс FeatureSelector: жадный прямой отбор, обратное исключение,
RFE и обёртку над sklearn SequentialFeatureSelector. Каждый метод даёт
упорядоченный список признаков (кроме sfs) и выбранный по нему набор.

Принципы:
    * Путь перебора проходится целиком (до max_features / min_features),
      а не обрывается на первом ухудшении: история хранит качество на
      каждом шаге, поэтому видно и как растёт кривая, и что происходит
      после оптимума. Оптимальный набор выбирается по готовой кривой.
    * Результат упорядочен: первые N элементов get_order(label) — набор
      из N признаков на пути перебора. Такой список можно передать
      в ModelValidator.evaluate_model_on_subsets для оценки на B.
    * forward/backward/sfs — фиксированный self.cv (кандидаты на шаге
      сравниваются на одинаковых фолдах), seed модели на фолде =
      fold_index, как в ModelValidator. rfe при n_runs > 1 — каждый
      прогон на своём разбиении cv_for_run(self.cv, run_index).
    * Параллелится только цикл по (кандидат, fold) — n_jobs.
      Многопоточные модели (XGBoost, LightGBM) нужно делать
      однопоточными через n_jobs=1 в kwargs фабрики.
    * Метрики — строки sklearn-scorer'ов (get_scorer), «больше — лучше».

Пример использования:
    X, y, feature_names, groups, classes = load_data("data.csv", "Y")
    fs = FeatureSelector(X, y, feature_names, random_state=42)
    factory = lambda s: make_xgb(s, n_jobs=1)
    fs.forward_selection(factory, scoring="f1_macro", label="fwd")
    fs.rfe(factory, n_runs=3, label="rfe")
    fs.get_history("fwd")              # кривая: n_features, f1_macro_mean, ...
    fs.get_selected_features("fwd")    # оптимальный набор
    fs.get_order("rfe")                # упорядоченный список -> ModelValidator
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import RFE, RFECV, SequentialFeatureSelector
from sklearn.metrics import get_scorer

from common import (
    ConfigurationError,
    check_cv,
    check_data,
    cv_for_run,
    make_folds,
)

# Настройка формата/уровня — на стороне пользователя, как в common.py.
logger = logging.getLogger("otbor.selection")
logger.addHandler(logging.NullHandler())


class FeatureSelector:
    """Методы перебора признаков с сохранением истории отбора.

    Каждый метод сохраняет результат под своим label:
        sequential_results[label] — dict:
            "method": str — 'forward' | 'backward' | 'rfe' | 'sfs';
            "scoring": str — метрика отбора;
            "order": list[str] — признаки в порядке пути перебора: первые
                N — набор из N признаков (у sfs — неупорядоченный набор);
            "n_selected": int — размер выбранного набора;
            "history": pd.DataFrame | None — кривая качества по шагам
                (у sfs — None);
            "scores": np.ndarray (n_steps, n_folds) | None — метрика
                в каждом фолде для строк history (у rfe — n_runs * n_folds
                столбцов).

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
        Число параллельных задач; -1 — все ядра.

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

        self.sequential_results = {}

    # -----------------------------------------------------------------------
    # Каркас
    # -----------------------------------------------------------------------

    @staticmethod
    def _run_fold(model_factory, X, y, columns, train_indices, val_indices,
                  seed, scoring):
        """Одна задача (кандидат, fold): обучение на columns, метрика на val.

        Выполняется в процессе-воркере joblib: всё нужное передаётся
        аргументами, результат — только через return.
        """
        X_cols = X[:, columns]
        model = model_factory(seed)
        model.fit(X_cols[train_indices], y[train_indices])
        return get_scorer(scoring)(model, X_cols[val_indices], y[val_indices])

    def _score_subsets(self, model_factory, subsets, folds, scoring):
        """Метрика каждого набора колонок в каждом фолде.

        Возвращает
        -------
        scores : np.ndarray, shape (len(subsets), n_folds)
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_fold)(model_factory, self.X, self.y, columns,
                                    train_indices, val_indices,
                                    fold_index, scoring)
            for columns in subsets
            for fold_index, (train_indices, val_indices) in enumerate(folds)
        )
        return np.array(results, dtype=np.float64).reshape(len(subsets),
                                                           len(folds))

    @staticmethod
    def _pick_n(n_features, mean, tolerance):
        """Наименьшее N, у которого mean >= max(mean) - tolerance.

        tolerance=0 -> N с наибольшим mean (при равенстве — меньшее).
        """
        n_features = np.asarray(n_features)
        within = mean >= mean.max() - tolerance
        return int(n_features[within].min())

    @staticmethod
    def _history(n_features, scores, scoring, features=None):
        """DataFrame истории: n_features, [feature], {scoring}_mean/_std."""
        history = pd.DataFrame({"n_features": n_features})
        if features is not None:
            history["feature"] = features
        history[f"{scoring}_mean"] = scores.mean(axis=1)
        history[f"{scoring}_std"] = scores.std(axis=1, ddof=1)
        return history

    def _check_label(self, label):
        if label in self.sequential_results:
            raise ConfigurationError(f"label={label!r} уже использован")

    def _get_result(self, label):
        """sequential_results[label] с понятной ошибкой при отсутствии."""
        if label not in self.sequential_results:
            raise ConfigurationError(
                f"Нет результатов для label={label!r}. "
                f"Доступны: {list(self.sequential_results)}"
            )
        return self.sequential_results[label]

    # -----------------------------------------------------------------------
    # Методы
    # -----------------------------------------------------------------------

    def forward_selection(self, model_factory, scoring="f1_macro",
                          max_features=None, tolerance=0.0, label="forward",
                          check_val_classes=True):
        """Жадный прямой отбор: на каждом шаге добавляется признак, дающий
        наибольшее среднее по фолдам значение метрики.

        Путь проходится до max_features признаков целиком. Выбранный набор
        — первые N признаков пути, где N — наименьшее число признаков
        с mean >= max(mean) - tolerance.

        Параметры
        ----------
        model_factory : callable
            seed -> модель, например lambda s: make_rf(s, n_estimators=100).
        scoring : str, default 'f1_macro'
            Строка sklearn-scorer'а.
        max_features : int | None, default None
            Длина пути; None -> все признаки.
        tolerance : float, default 0.0
            Допуск (в единицах метрики) при выборе N: меньший набор
            предпочитается, если хуже лучшего не более чем на tolerance.
        label : str, default 'forward'
            Уникальный label результата.
        check_val_classes : bool, default True
            Требовать все классы в каждом val-фолде (см. make_folds).

        Возвращает
        -------
        self

        История (get_history): строка на шаг; n_features — размер набора
        после шага, feature — добавленный признак.
        """
        self._check_label(label)
        n_total = len(self.feature_names)
        if max_features is None:
            max_features = n_total
        if not 1 <= max_features <= n_total:
            raise ConfigurationError(
                f"max_features={max_features} вне диапазона 1..{n_total}"
            )

        folds = make_folds(self.cv, self.X, self.y, self.groups,
                           check_val_classes=check_val_classes)
        selected, remaining = [], list(range(n_total))
        step_scores = []

        while len(selected) < max_features:
            candidates = self._score_subsets(
                model_factory, [selected + [j] for j in remaining],
                folds, scoring,
            )
            best = int(np.argmax(candidates.mean(axis=1)))
            selected.append(remaining.pop(best))
            step_scores.append(candidates[best])
            logger.info(
                f"{label}: {len(selected)}/{max_features} "
                f"+{self.feature_names[selected[-1]]}, "
                f"{scoring}={candidates[best].mean():.4f}"
            )

        order = [self.feature_names[j] for j in selected]
        scores = np.array(step_scores)
        ns = list(range(1, max_features + 1))
        self.sequential_results[label] = {
            "method": "forward",
            "scoring": scoring,
            "order": order,
            "n_selected": self._pick_n(ns, scores.mean(axis=1), tolerance),
            "history": self._history(ns, scores, scoring, order),
            "scores": scores,
        }
        logger.info(f"{label}: готово")
        return self

    def backward_elimination(self, model_factory, scoring="f1_macro",
                             min_features=1, tolerance=0.0, label="backward",
                             check_val_classes=True):
        """Жадное обратное исключение: на каждом шаге удаляется признак, без
        которого среднее по фолдам значение метрики наибольшее.

        Путь проходится от всех признаков до min_features целиком. Выбор N
        — как в forward_selection.

        Параметры
        ----------
        model_factory : callable
            seed -> модель.
        scoring : str, default 'f1_macro'
            Строка sklearn-scorer'а.
        min_features : int, default 1
            Размер набора в конце пути.
        tolerance : float, default 0.0
            Допуск при выборе N (см. forward_selection).
        label : str, default 'backward'
            Уникальный label результата.
        check_val_classes : bool, default True
            Требовать все классы в каждом val-фолде (см. make_folds).

        Возвращает
        -------
        self

        История (get_history): первая строка — все признаки (feature=None),
        далее строка на шаг; n_features — размер набора после шага,
        feature — удалённый признак. Порядок get_order: оставшиеся
        в конце пути, затем удалённые от последнего к первому.
        """
        self._check_label(label)
        n_total = len(self.feature_names)
        if not 1 <= min_features <= n_total:
            raise ConfigurationError(
                f"min_features={min_features} вне диапазона 1..{n_total}"
            )

        folds = make_folds(self.cv, self.X, self.y, self.groups,
                           check_val_classes=check_val_classes)
        current = list(range(n_total))
        removed = []
        step_scores = [self._score_subsets(model_factory, [current],
                                           folds, scoring)[0]]

        while len(current) > min_features:
            candidates = self._score_subsets(
                model_factory,
                [current[:i] + current[i + 1:] for i in range(len(current))],
                folds, scoring,
            )
            best = int(np.argmax(candidates.mean(axis=1)))
            removed.append(current.pop(best))
            step_scores.append(candidates[best])
            logger.info(
                f"{label}: {len(current)} признаков, "
                f"-{self.feature_names[removed[-1]]}, "
                f"{scoring}={candidates[best].mean():.4f}"
            )

        # Набор из N признаков на пути = order[:N].
        order = [self.feature_names[j] for j in current + removed[::-1]]
        scores = np.array(step_scores)
        ns = list(range(n_total, min_features - 1, -1))
        features = [None] + [self.feature_names[j] for j in removed]
        self.sequential_results[label] = {
            "method": "backward",
            "scoring": scoring,
            "order": order,
            "n_selected": self._pick_n(ns, scores.mean(axis=1), tolerance),
            "history": self._history(ns, scores, scoring, features),
            "scores": scores,
        }
        logger.info(f"{label}: готово")
        return self

    def rfe(self, model_factory, n_features_to_select=None, step=1, n_runs=1,
            scoring="f1_macro", label="rfe", check_val_classes=True,
            **kwargs):
        """Recursive Feature Elimination: исключение признаков с наименьшей
        важностью модели (coef_ / feature_importances_).

        Кривая качества — RFECV в каждом прогоне (прогон — своё разбиение
        cv_for_run(self.cv, run_index), seed модели = run_index); метрика
        усредняется по всем прогонам и фолдам. Порядок признаков — RFE
        до одного признака на всех данных (seed = 0). Выбранный набор —
        первые N признаков порядка.

        Параметры
        ----------
        model_factory : callable
            seed -> модель с coef_ или feature_importances_. Для Pipeline
            из фабрик (make_logreg) нужен importance_getter в kwargs,
            например importance_getter="named_steps.model.coef_".
        n_features_to_select : int | None, default None
            Размер выбранного набора; None -> N с наибольшим mean кривой
            (при равенстве — меньшее).
        step : int, default 1
            Сколько признаков исключается за шаг.
        n_runs : int, default 1
            Число прогонов CV.
        scoring : str, default 'f1_macro'
            Строка sklearn-scorer'а для кривой.
        label : str, default 'rfe'
            Уникальный label результата.
        check_val_classes : bool, default True
            Требовать все классы в каждом val-фолде (см. make_folds).
        **kwargs
            Передаются в RFECV и RFE (например, importance_getter).

        Возвращает
        -------
        self

        История (get_history): n_features по возрастанию, без колонки
        feature. При step > 1 признаки, исключённые за один шаг, в
        get_order идут в порядке колонок X.
        """
        self._check_label(label)
        n_total = len(self.feature_names)
        if n_features_to_select is not None and \
                not 1 <= n_features_to_select <= n_total:
            raise ConfigurationError(
                f"n_features_to_select={n_features_to_select} вне "
                f"диапазона 1..{n_total}"
            )

        run_scores = []   # по прогону: (n_points, n_folds)
        for run_index in range(n_runs):
            folds = make_folds(cv_for_run(self.cv, run_index), self.X,
                               self.y, self.groups,
                               check_val_classes=check_val_classes)
            rfecv = RFECV(model_factory(run_index), step=step, cv=folds,
                          scoring=scoring, min_features_to_select=1,
                          n_jobs=self.n_jobs, **kwargs)
            rfecv.fit(self.X, self.y)
            res = rfecv.cv_results_
            sort = np.argsort(res["n_features"])
            ns = [int(n) for n in res["n_features"][sort]]
            run_scores.append(np.column_stack(
                [res[f"split{i}_test_score"][sort] for i in range(len(folds))]
            ))
            logger.info(f"{label}: прогон {run_index + 1}/{n_runs}")
        scores = np.hstack(run_scores)

        ranker = RFE(model_factory(0), n_features_to_select=1, step=step,
                     **kwargs).fit(self.X, self.y)
        order = [self.feature_names[j]
                 for j in np.argsort(ranker.ranking_, kind="stable")]

        if n_features_to_select is None:
            n_features_to_select = self._pick_n(ns, scores.mean(axis=1), 0.0)
        self.sequential_results[label] = {
            "method": "rfe",
            "scoring": scoring,
            "order": order,
            "n_selected": n_features_to_select,
            "history": self._history(ns, scores, scoring),
            "scores": scores,
        }
        logger.info(f"{label}: готово")
        return self

    def sequential_feature_selector(self, model_factory,
                                    n_features_to_select="auto",
                                    direction="forward", scoring="f1_macro",
                                    label="sfs", check_val_classes=True,
                                    **kwargs):
        """Обёртка над sklearn SequentialFeatureSelector на фолдах self.cv.

        В отличие от forward_selection / backward_elimination, sklearn
        останавливается на n_features_to_select и не хранит историю:
        результат — только итоговый неупорядоченный набор. Модель одна
        на все фолды (sklearn клонирует её), seed = 0.

        Параметры
        ----------
        model_factory : callable
            seed -> модель.
        n_features_to_select : int | float | 'auto', default 'auto'
            См. SequentialFeatureSelector; 'auto' без tol -> половина
            признаков, с tol (в kwargs) — остановка по приросту метрики.
        direction : {'forward', 'backward'}, default 'forward'
        scoring : str, default 'f1_macro'
            Строка sklearn-scorer'а.
        label : str, default 'sfs'
            Уникальный label результата.
        check_val_classes : bool, default True
            Требовать все классы в каждом val-фолде (см. make_folds).
        **kwargs
            Передаются в SequentialFeatureSelector (например, tol).

        Возвращает
        -------
        self
        """
        self._check_label(label)
        folds = make_folds(self.cv, self.X, self.y, self.groups,
                           check_val_classes=check_val_classes)
        sfs = SequentialFeatureSelector(
            model_factory(0), n_features_to_select=n_features_to_select,
            direction=direction, scoring=scoring, cv=folds,
            n_jobs=self.n_jobs, **kwargs,
        ).fit(self.X, self.y)
        selected = [self.feature_names[j]
                    for j in np.flatnonzero(sfs.get_support())]
        self.sequential_results[label] = {
            "method": "sfs",
            "scoring": scoring,
            "order": selected,
            "n_selected": len(selected),
            "history": None,
            "scores": None,
        }
        logger.info(f"{label}: готово")
        return self

    # -----------------------------------------------------------------------
    # Геттеры
    # -----------------------------------------------------------------------

    def get_selected_features(self, label):
        """Выбранный набор признаков: первые n_selected элементов порядка.

        Параметры
        ----------
        label : str

        Возвращает
        -------
        subset : list[str]
        """
        result = self._get_result(label)
        return result["order"][:result["n_selected"]]

    def get_order(self, label):
        """Признаки в порядке пути перебора: первые N — набор из N.

        Подходит как ranking_or_subset для ModelValidator.

        Параметры
        ----------
        label : str

        Возвращает
        -------
        order : list[str]

        Исключения
        ----------
        ConfigurationError
            У метода нет порядка (sfs).
        """
        result = self._get_result(label)
        if result["history"] is None:
            raise ConfigurationError(
                f"{label}: у метода {result['method']!r} нет порядка, "
                "только набор (get_selected_features)"
            )
        return list(result["order"])

    def get_history(self, label):
        """История отбора: кривая качества по шагам.

        Параметры
        ----------
        label : str

        Возвращает
        -------
        history : pd.DataFrame
            Колонки n_features, feature (кроме rfe), {scoring}_mean,
            {scoring}_std (std по фолдам, ddof=1).

        Исключения
        ----------
        ConfigurationError
            У метода нет истории (sfs).
        """
        result = self._get_result(label)
        if result["history"] is None:
            raise ConfigurationError(
                f"{label}: у метода {result['method']!r} нет истории"
            )
        return result["history"].copy()
