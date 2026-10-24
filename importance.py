"""
importance.py — оценка информативности признаков (ранжирование).

Содержит класс ImportanceEvaluator:
    * каркас _stable_cv и задача одного фолда _run_fold;
    * функции важности за фолд (_mi_fold, _anova_f_fold, ...) — staticmethod;
    * методы compute_* и геттеры.

Принципы:
    * Все методы считаются по одной схеме: n_runs прогонов CV, в каждом
      прогоне своё разбиение cv_for_run(self.cv, run_index). Одинаковый
      run_index даёт одинаковые фолды во всех методах, поэтому ранжирования
      разных методов парные.
    * Seed модели/метода на фолде: run_index * 1000 + fold_index.
    * Параллелится только цикл по (run, fold) — n_jobs. Всё, что внутри
      фолда, однопоточное; многопоточные модели (XGBoost, LightGBM)
      нужно делать однопоточными через n_jobs=1 в kwargs фабрики.
    * Метод = функция важности за фолд + общий цикл _stable_cv.
      Функция отвечает на вопрос «что посчитать на фолде», цикл — «на
      каких фолдах и как усреднить» (аналогично фабрикам make_* в common).

Пример использования:
    X, y, feature_names, groups, classes = load_data("data.csv", "Y")
    ev = ImportanceEvaluator(X, y, feature_names, n_runs=5, random_state=42)
    ev.compute_mi("mi").compute_anova_f("anova_f")
    ev.compute_mdi(lambda s: make_rf(s, n_estimators=100), "mdi_rf")
    ev.compute_l1_importance(lambda s: make_logreg(s, C=0.1, l1_ratio=1), "l1")
    ev.compute_pi(lambda s: make_rf(s), "pi_rf",
                  scoring=["roc_auc_ovr", "balanced_accuracy"])
    ev.compute_shap(lambda s: make_rf(s), "shap_rf")
    ev.get_ranking("pi_rf_roc_auc_ovr")
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from common import (
    ConfigurationError,
    check_cv,
    check_data,
    cv_for_run,
    get_filter_func,
    get_predict_fn,
    make_folds,
)

# Настройка формата/уровня — на стороне пользователя, как в common.py.
# Логировать можно только в основном процессе: в процессах-воркерах joblib
# настройки logging пользователя нет, и сообщения теряются.
logger = logging.getLogger("otbor.importance")
logger.addHandler(logging.NullHandler())


class ImportanceEvaluator:
    """Оценка информативности признаков набором методов на общих фолдах.

    Каждый compute_* сохраняет результат под своим label:
        importance_results[label] — DataFrame с колонками
            feature, {label}_mean, {label}_std (по прогонам);
        raw_importance[label] — np.ndarray (n_runs, n_features): важность,
            усреднённая по фолдам внутри каждого прогона.

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
    n_runs : int, default 5
        Число прогонов CV по умолчанию для всех методов.
    random_state : int, default 42
        Seed дефолтного сплиттера (при cv=None).
    n_jobs : int, default -1
        Число параллельных задач (run, fold); -1 — все ядра.

    Исключения
    ----------
    DataValidationError
        Несогласованные размеры X, y, groups, feature_names.
    ConfigurationError
        Некорректный cv.
    """

    def __init__(self, X, y, feature_names, groups=None, cv=None, n_runs=5,
                 random_state=42, n_jobs=-1):
        X, y, feature_names, groups = check_data(X, y, feature_names, groups)

        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.groups = groups
        self.cv = check_cv(cv, random_state)
        self.n_runs = n_runs
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.importance_results = {}
        self.raw_importance = {}

    # -----------------------------------------------------------------------
    # Каркас
    # -----------------------------------------------------------------------

    @staticmethod
    def _run_fold(importance_fn, model_factory, X, y, train_indices,
                  val_indices, seed, fn_kwargs):
        """Одна задача (run, fold) параллельного цикла _stable_cv.

        Выполняется в отдельном процессе-воркере joblib, поэтому всё нужное
        передаётся аргументами, а результат возвращается только через return.
        train_indices, val_indices — массивы номеров строк.
        """
        X_train, y_train = X[train_indices], y[train_indices]
        if model_factory is None:
            return importance_fn(X_train, y_train, seed=seed, **fn_kwargs)
        model = model_factory(seed)
        model.fit(X_train, y_train)
        return importance_fn(model, X[val_indices], y[val_indices],
                             seed=seed, X_train=X_train, **fn_kwargs)

    def _stable_cv(self, importance_fn, label, model_factory=None,
                   multi_output=False, n_runs=None, check_val_classes=False,
                   **fn_kwargs):
        """Каркас всех compute_*: n_runs прогонов CV x фолды.

        Параметры
        ----------
        importance_fn : callable
            Функция важности за фолд (см. раздел «Функции важности за фолд»).
        label : str
            Под ним сохраняется результат; при multi_output — префикс
            под-лейблов f"{label}_{metric}". Уникальность под-лейблов
            проверяет вызывающий метод.
        model_factory : callable | None, default None
            None — режим без модели: importance_fn считается на train-части.
            Иначе на фолде обучается model_factory(seed).
        multi_output : bool, default False
            importance_fn возвращает dict {metric: np.ndarray}.
        n_runs : int | None, default None
            None -> self.n_runs.
        check_val_classes : bool, default False
            Передаётся в make_folds: требовать все классы в val-фолдах.
        **fn_kwargs
            Передаются в importance_fn.

        Возвращает
        -------
        labels : list[str]
        """
        if n_runs is None:
            n_runs = self.n_runs
        if not multi_output and label in self.importance_results:
            raise ConfigurationError(f"label={label!r} уже использован")

        # 1. Список задач: все фолды всех прогонов.
        # train_indices, val_indices — массивы номеров строк отдельного фолда;
        tasks = []   # (run_index, train_indices, val_indices, seed)

        for run_index in range(n_runs):
            folds = make_folds(cv_for_run(self.cv, run_index), self.X, self.y,
                               self.groups,
                               check_val_classes=check_val_classes)
            for fold_index, (train_indices, val_indices) in enumerate(folds):
                tasks.append((run_index, train_indices, val_indices,
                              run_index * 1000 + fold_index))

        # 2. Параллельный расчёт: по одной задаче _run_fold на фолд.
        # results = []
        # for _, train_indices, val_indices, seed in tasks:
        #   results.append(self._run_fold(...))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_fold)(importance_fn, model_factory,
                                    self.X, self.y,
                                    train_indices, val_indices,
                                    seed, fn_kwargs)
            for _, train_indices, val_indices, seed in tasks
        )

        task_run = np.array([t[0] for t in tasks]) # task_run[k] — номер прогона k-й задачи.

        # 3. Раскладка по лейблам: {label: [вектор за каждый фолд]}.
        if multi_output:
            per_label = {f"{label}_{metric}": [r[metric] for r in results]
                         for metric in results[0]}
        else:
            per_label = {label: results}

        # 4. Агрегация важности: фолды -> прогон (raw), прогоны -> mean/std.
        for lbl, fold_vectors in per_label.items():
            fold_vectors = np.asarray(fold_vectors, dtype=np.float64)
            nan_mask = np.isnan(fold_vectors)
            if nan_mask.any():
                logger.warning(
                    f"{lbl}: Получено значение важности NaN у "
                    f"{int(nan_mask.any(axis=0).sum())} признак(ов) -> 0"
                )
                fold_vectors = np.where(nan_mask, 0.0, fold_vectors)

            raw = np.stack([fold_vectors[task_run == r].mean(axis=0)
                            for r in range(n_runs)])
            
            if n_runs > 1:
                std = raw.std(axis=0, ddof=1) # ddof=1 — несмещенная оценка дисперсии;
            else:
                std = np.full(raw.shape[1], np.nan)

            self.raw_importance[lbl] = raw
            self.importance_results[lbl] = pd.DataFrame({
                "feature": self.feature_names,
                f"{lbl}_mean": raw.mean(axis=0),
                f"{lbl}_std": std,
            })
        logger.info(f"{label}: готово")
        return list(per_label)

    # -----------------------------------------------------------------------
    # Функции важности за фолд
    # -----------------------------------------------------------------------
    # Режим без модели (filter-методы): fn(X_train, y_train, seed=..., **kw)
    #     -> np.ndarray (n_features,)
    # Режим с моделью: fn(model, X_val, y_val, seed=..., X_train=..., **kw)
    #     -> np.ndarray (n_features,) или dict {metric: np.ndarray}
    #        (multi_output)
    # staticmethod: self не нужен, а в воркер joblib не копируется весь
    # объект. Выполняются в воркерах, поэтому не логируют: о проблемах
    # сообщают через возвращаемое значение (например, NaN), а _stable_cv
    # их обрабатывает.

    @staticmethod
    def _mi_fold(X_train, y_train, seed, n_neighbors=3, **kwargs):
        """Mutual information каждого признака с классом на train-части фолда.

        sklearn при n_neighbors >= размера класса молча уменьшает k для этого
        класса, и оценки по классам становятся несопоставимыми — поэтому
        ошибка, а не тихая подмена.
        """
        counts = np.bincount(np.asarray(y_train))
        n_min = int(counts[counts > 0].min())

        if n_neighbors >= n_min:
            raise ConfigurationError(
                f"MI: n_neighbors={n_neighbors} должно быть меньше размера "
                f"самого малого класса в train-фолде ({n_min})"
            )
        return get_filter_func("mi")(
            X_train, y_train,
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=seed,
            **kwargs,
        )

    @staticmethod
    def _anova_f_fold(X_train, y_train, seed=None):
        """F-статистика ANOVA каждого признака на train-части фолда.

        p-value отбрасывается. Для признака, константного внутри всех классов,
        F = NaN; на 0 его заменяет _stable_cv. seed не используется:
        f_classif детерминирован.
        """
        F, _ = get_filter_func("anova_f")(X_train, y_train)
        return F

    @staticmethod
    def _get_estimator(model):
        """Сама модель: последний шаг Pipeline из фабрик, иначе model."""
        return model[-1] if hasattr(model, "named_steps") else model


    @staticmethod
    def _mdi_fold(model, X_val, y_val, seed=None, X_train=None):
        """MDI обученной модели (feature_importances_); val не используется."""
        return ImportanceEvaluator._get_estimator(model).feature_importances_

    @staticmethod
    def _l1_fold(model, X_val, y_val, seed=None, X_train=None, agg="sum"):
        """Модули коэффициентов, агрегированные по классам; val не нужен.

        coef_ у LogisticRegression имеет форму (K, p), при двух классах —
        (1, p). 
        atleast_2d нужен для регрессии: там coef_
        одномерный (p,), и без него sum(axis=0) свернул бы признаки в число.
        """
        estimator = ImportanceEvaluator._get_estimator(model)
        coef = np.abs(np.atleast_2d(estimator.coef_))
        return coef.sum(axis=0) if agg == "sum" else coef.max(axis=0)

    @staticmethod
    def _pi_fold(model, X_val, y_val, seed, X_train=None, scoring=None,
                 n_perm_reps=30, **kwargs):
        """Permutation importance на val-части фолда по всем метрикам сразу.

        Перестановки делаются один раз, на них считаются все метрики.
        n_jobs=1: параллелится внешний цикл по фолдам.

        Возвращает
        -------
        dict
            {metric: importances_mean}.
        """
        result = permutation_importance(
            model, X_val, y_val,
            scoring=scoring,
            n_repeats=n_perm_reps,
            random_state=seed,
            n_jobs=1,
            **kwargs,
        )
        return {metric: r.importances_mean for metric, r in result.items()}

    @staticmethod
    def _shap_fold(model, X_val, y_val, seed, X_train, explainer,
                   model_output, n_background, max_evals):
        """SHAP-важность на val-части фолда: sum_k mean_i |phi_ijk|.

        Интервенционный режим, фон — случайная подвыборка train-части
        (seed). y_val не используется. 
        explainer уже выбран и проверен в compute_shap.
        Модуль берётся до суммы по классам: в пространстве вероятностей
        вклады по классам в сумме дают 0.
        """
        rng = np.random.default_rng(seed)
        n_bg = min(n_background, len(X_train))
        background = X_train[rng.choice(len(X_train), n_bg, replace=False)]

        if explainer == "tree":
            # Округление до float32 (с возвратом во float64): shap хранит
            # пороги LGBM во float32, и на исходных float64 значения рядом с
            # порогом уходят не в ту ветку (сумма вкладов расходится с
            # выходом модели до 0.5). На округлённых данных сумма вкладов
            # совпадает с выходом модели на исходных данных (~1e-7), а для
            # RF/ET/XGB округление ничего не меняет. Встроенная проверка
            # shap сравнивает с моделью на округлённых данных и ложно
            # падает, поэтому check_additivity=False.
            phi = shap.TreeExplainer(
                model,
                data=background.astype(np.float32).astype(np.float64),
                feature_perturbation="interventional",
                model_output=model_output,
            )(X_val.astype(np.float32).astype(np.float64),
              check_additivity=False).values
        elif explainer == "linear":
            # Вклады считаются в координатах после скейлера: он сдвигает и
            # масштабирует каждый признак отдельно, и для линейной модели
            # вклады от этого не меняются.
            scaler = model.named_steps["scaler"]
            masker = shap.maskers.Independent(scaler.transform(background),
                                              max_samples=n_bg)
            phi = shap.LinearExplainer(
                ImportanceEvaluator._get_estimator(model), masker,
            )(scaler.transform(X_val)).values
        else:
            masker = shap.maskers.Independent(background, max_samples=n_bg)
            phi = shap.PermutationExplainer(
                get_predict_fn(model), masker, seed=seed,
            )(X_val, max_evals=max_evals, silent=True).values

        # Один выход (логиты бинарной задачи): (m, p) -> (m, p, 1).
        if phi.ndim == 2:
            phi = phi[:, :, None]
        return np.abs(phi).mean(axis=0).sum(axis=1)

    # -----------------------------------------------------------------------
    # Filter methods
    # -----------------------------------------------------------------------

    def compute_mi(self, label, n_runs=None, n_neighbors=3, **kwargs):
        """Mutual information признака с классом (mutual_info_classif).

        Оценка по ближайшим соседям, discrete_features=False; каждый
        признак оценивается отдельно. Считается на train-части фолдов.

        Параметры
        ----------
        label : str
            Уникальный label результата.
        n_runs : int | None, default None
            None -> self.n_runs.
        n_neighbors : int, default 3
            Число соседей k; должно быть меньше размера самого малого
            класса в каждом train-фолде.
        **kwargs
            Остальные параметры mutual_info_classif (например, copy);
            не должны дублировать явные аргументы и random_state.

        Возвращает
        -------
        self
        """
        self._stable_cv(self._mi_fold, label, n_runs=n_runs,
                        n_neighbors=n_neighbors, **kwargs)
        return self

    def compute_anova_f(self, label, n_runs=None):
        """F-статистика однофакторного ANOVA (f_classif).

        Межклассовый разброс признака относительно внутриклассового;
        p-value отбрасывается. Считается на train-части фолдов.

        Параметры
        ----------
        label : str
            Уникальный label результата.
        n_runs : int | None, default None
            None -> self.n_runs.

        Возвращает
        -------
        self
        """
        self._stable_cv(self._anova_f_fold, label, n_runs=n_runs)
        return self

    # -----------------------------------------------------------------------
    # Embedded methods
    # -----------------------------------------------------------------------

    def compute_mdi(self, model_factory, label, n_runs=None):
        """Mean Decrease in Impurity древесной модели (feature_importances_).

        Модель обучается на train-части фолда, val-часть не используется.

        Параметры
        ----------
        model_factory : callable
            seed -> модель с feature_importances_ (make_rf, make_et,
            make_xgb, make_lgbm), например lambda s: make_rf(s, n_estimators=100).
        label : str
            Уникальный label результата.
        n_runs : int | None, default None
            None -> self.n_runs.

        Возвращает
        -------
        self
        """
        # До обучения feature_importances_ недоступен у экземпляра,
        # поэтому проверяется класс. model_factory(0) — только для проверки.
        estimator = self._get_estimator(model_factory(0))
        if not hasattr(type(estimator), "feature_importances_"):
            raise ConfigurationError(
                f"MDI: у {type(estimator).__name__} нет feature_importances_ "
            )
        self._stable_cv(self._mdi_fold, label, model_factory=model_factory,
                        n_runs=n_runs)
        return self

    def compute_l1_importance(self, model_factory, label, agg="sum",
                              n_runs=None):
        """Важность по модулям коэффициентов L1-логистической регрессии.

        Модель обучается на train-части фолда, val-часть не используется.
        Признаки стандартизованы скейлером пайплайна, поэтому модули
        коэффициентов сопоставимы между признаками.

        Параметры
        ----------
        model_factory : callable
            Фабрика на основе make_logreg с l1_ratio > 0, например
            lambda s: make_logreg(s, C=0.1, l1_ratio=1). C фиксирован
            для всех фолдов. 
            Elastic-net (0 < l1_ratio < 1) допускается.
        label : str
            Уникальный label результата.
        agg : {'sum', 'max'}, default 'sum'
            Агрегация по классам: sum_k |beta_kj| или max_k |beta_kj|.
        n_runs : int | None, default None
            None -> self.n_runs.

        Возвращает
        -------
        self
        """
        if agg not in ("sum", "max"):
            raise ConfigurationError(
                f"agg должен быть 'sum' или 'max', получено {agg!r}"
            )
        # model_factory(0) — только для проверки параметров.
        if self._get_estimator(model_factory(0)).l1_ratio == 0:
            raise ConfigurationError(
                "L1: в модели l1_ratio=0, коэффициенты не обнуляются."
                "Требуется задать l1_ratio > 0 в make_logreg."
            )
        self._stable_cv(self._l1_fold, label, model_factory=model_factory,
                        n_runs=n_runs, agg=agg)
        return self

    # -----------------------------------------------------------------------
    # Wrapper/Post-hoc methods
    # -----------------------------------------------------------------------

    def compute_pi(self, model_factory, label, scoring=("roc_auc_ovr",),
                   n_perm_reps=30, n_runs=None, check_val_classes=True,
                   **kwargs):
        """Permutation importance (permutation_importance) на val-части фолда.

        Важность — падение метрики при перестановке признака
        (importances_mean); отрицательные значения не обрезаются. Результат
        по каждой метрике — под под-лейблом f"{label}_{metric}".

        Параметры
        ----------
        model_factory : callable
            seed -> модель.
        label : str
            Префикс под-лейблов.
        scoring : str | list[str], default ('roc_auc_ovr',)
            Строки sklearn-scorer'ов; строка заворачивается в список.
            Для многоклассового случая: roc_auc_ovr, balanced_accuracy,
            f1_macro, neg_log_loss (f1 и roc_auc падают).
        n_perm_reps : int, default 30
            Число перестановок каждого признака.
        n_runs : int | None, default None
            None -> self.n_runs.
        check_val_classes : bool, default True
            Требовать все классы в каждом val-фолде (нужно для roc_auc_ovr
            и neg_log_loss). При LOGO, где val — одна группа (скважина),
            можно отключить и считать по balanced_accuracy / f1_macro.
        **kwargs
            Остальные параметры permutation_importance (например,
            max_samples); не должны дублировать явные аргументы, n_jobs
            и random_state.

        Возвращает
        -------
        self
        """
        # Строка -> список: со строкой sklearn вернёт Bunch, а не dict.
        scoring = [scoring] if isinstance(scoring, str) else list(scoring)
        for metric in scoring:
            sub_label = f"{label}_{metric}"
            if sub_label in self.importance_results:
                raise ConfigurationError(
                    f"label={sub_label!r} уже использован. Задай другой label."
                )
        self._stable_cv(self._pi_fold, label, model_factory=model_factory,
                        multi_output=True, n_runs=n_runs,
                        check_val_classes=check_val_classes,
                        scoring=scoring, n_perm_reps=n_perm_reps, **kwargs)
        return self

    def compute_shap(self, model_factory, label, explainer="auto",
                     model_output="probability", n_background=100,
                     max_evals=500, n_runs=None):
        """SHAP-важность (интервенционный режим) на val-части фолда.

        Модель обучается на train-части фолда, объясняется val-часть, фон —
        подвыборка train-части. Важность признака j — sum_k mean_i
        |phi_ijk|.

        Объяснители:
            'tree'        — TreeExplainer, точно; RF/ET при 'probability',
                            XGB/LGBM при 'raw' (вероятности бустингов
                            shap 0.54 объясняет неверно);
            'permutation' — PermutationExplainer по predict_proba, оценка;
                            любая модель, только model_output='probability';
            'linear'      — LinearExplainer, точно; только LogReg и
                            model_output='raw';
            'auto'        — 'tree', если подходит, иначе 'linear' при 'raw'
                            и 'permutation' при 'probability'.

        Параметры
        ----------
        model_factory : callable
            seed -> модель.
        label : str
            Уникальный label результата.
        explainer : {'auto', 'tree', 'permutation', 'linear'}, default 'auto'
        model_output : {'probability', 'raw'}, default 'probability'
            'probability' — вероятности, одна шкала для всех моделей;
            'raw' — логиты (только XGB, LGBM, LogReg).
        n_background : int, default 100
            Размер фона (берётся не больше размера train-части).
        max_evals : int, default 500
            Бюджет вызовов модели на объект для PermutationExplainer;
            не меньше 2 * n_features + 1 (одна перестановка вперёд-назад).
        n_runs : int | None, default None
            None -> self.n_runs.

        Возвращает
        -------
        self
        """
        if explainer not in ("auto", "tree", "permutation", "linear"):
            raise ConfigurationError(
                "explainer должен быть 'auto', 'tree', 'permutation' или "
                f"'linear', получено {explainer!r}"
            )
        if model_output not in ("probability", "raw"):
            raise ConfigurationError(
                "model_output должен быть 'probability' или 'raw', "
                f"получено {model_output!r}"
            )

        # model_factory(0) — только для выбора и проверки объяснителя.
        # Дерево — модель с feature_importances_ без Pipeline:
        # TreeExplainer принимает только саму модель.
        model = model_factory(0)
        estimator = self._get_estimator(model)
        is_tree = (not hasattr(model, "named_steps")
                   and hasattr(type(estimator), "feature_importances_"))
        # RF/ET — деревья sklearn, XGB/LGBM — бустинги из своих библиотек.
        is_sklearn_tree = (is_tree and
                           type(estimator).__module__.startswith("sklearn"))
        name = type(estimator).__name__

        # У деревьев sklearn выход — уже вероятности, логитов нет.
        if model_output == "raw" and is_sklearn_tree:
            raise ConfigurationError(
                f"SHAP: model_output='raw' не подходит для {name}: "
                "логиты есть только у XGB, LGBM и LogReg"
            )
        if explainer == "auto":
            if is_tree and (model_output == "probability") == is_sklearn_tree:
                explainer = "tree"
            elif model_output == "raw":
                explainer = "linear"
            else:
                explainer = "permutation"

        if explainer == "tree" and not is_tree:
            raise ConfigurationError(
                f"SHAP: explainer='tree' не подходит для {name} "
                "(нужна RF/ET/XGB/LGBM без Pipeline)"
            )
        # shap 0.54: вероятности многоклассового XGB TreeExplainer не
        # поддерживает, а у LGBM молча считает неверно (сумма вкладов
        # расходится с predict_proba). Логиты обоих — точно.
        if explainer == "tree" and model_output == "probability" \
                and not is_sklearn_tree:
            raise ConfigurationError(
                f"SHAP: TreeExplainer неверно объясняет вероятности {name}. "
                "Используй explainer='permutation' или model_output='raw'"
            )
        if explainer == "linear" and not (
                model_output == "raw"
                and isinstance(estimator, LogisticRegression)):
            raise ConfigurationError(
                f"SHAP: explainer='linear' — только для LogReg и "
                f"model_output='raw' (получено {name}, {model_output!r})"
            )
        if explainer == "permutation":
            if model_output == "raw":
                raise ConfigurationError(
                    "SHAP: explainer='permutation' объясняет вероятности; "
                    "для model_output='raw' нужен 'tree' (XGB, LGBM) "
                    "или 'linear' (LogReg)"
                )
            min_evals = 2 * self.X.shape[1] + 1
            if max_evals < min_evals:
                raise ConfigurationError(
                    f"SHAP: max_evals={max_evals} меньше одной перестановки "
                    f"(2 * n_features + 1 = {min_evals})"
                )

        self._stable_cv(self._shap_fold, label, model_factory=model_factory,
                        n_runs=n_runs, explainer=explainer,
                        model_output=model_output, n_background=n_background,
                        max_evals=max_evals)
        return self

    # -----------------------------------------------------------------------
    # Геттеры
    # -----------------------------------------------------------------------

    def get_ranking(self, label):
        """Ранжирование признаков по убыванию {label}_mean.

        Параметры
        ----------
        label : str
            Label метода; для PI — под-лейбл f"{label}_{metric}".

        Возвращает
        -------
        ranking : pd.DataFrame
            Колонки feature, {label}_mean, {label}_std; индекс 0..p-1.
        """
        if label not in self.importance_results:
            raise ConfigurationError(
                f"Нет результатов для label={label!r}. "
                f"Доступны: {list(self.importance_results)}"
            )
        return (self.importance_results[label]
                .sort_values(f"{label}_mean", ascending=False, kind="stable")
                .reset_index(drop=True))

    def get_all_rankings(self):
        """Все ранжирования: {label: pd.DataFrame} (см. get_ranking)."""
        return {label: self.get_ranking(label)
                for label in self.importance_results}
