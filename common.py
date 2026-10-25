"""
common.py — общие утилиты проекта otbor (отбор признаков).

Содержит:
    1. Исключения проекта (DataValidationError, ConfigurationError)
       и проверку типа задачи (check_task)
    2. Фабрики моделей:
       классификация — make_rf, make_xgb, make_lgbm, make_et,
       make_knn, make_logreg, make_svc, make_mlp;
       регрессия — make_rf_reg, make_xgb_reg, make_lgbm_reg, make_et_reg,
       make_knn_reg, make_linreg, make_svr, make_mlp_reg
    3. Утилиты кросс-валидации (check_cv, cv_for_run, make_folds)
    4. Хелперы методов важности и метрик (get_filter_func,
       check_mi_neighbors, get_predict_fn, get_predict_output,
       get_default_scoring)
    5. Загрузчик данных (load_data) и проверка массивов (check_data)

Принципы:
    * Seed всегда задаётся явно при создании модели — воспроизводимость.
    * Все фабрики принимают **kwargs — любой гиперпараметр можно
      переопределить без правки кода фабрики.
    * Данные на входе в пайплайн должны быть "чистыми": без NaN
      и без нечисловых колонок. Проблемы ловятся в load_data
      с понятными сообщениями, а не глубоко внутри fit().
    * xgboost и lightgbm импортируются внутри фабрик,
      чтобы common.py работал даже если они не установлены.
    * Тип задачи (task='classification' | 'regression') влияет только на
      хелперы разделов 3–5 этого модуля: ветки по задаче — здесь, а не
      в методах классов. Фабрики по task не ветвятся: для регрессии —
      отдельные make_*_reg.
# 
Пример использования:
    X, y, feature_names, groups, classes = load_data(
        "data.csv", target_col="Y", group_col="well_id",
        cols_to_drop=["id", "date"],
    )
    model = make_rf(seed=42, n_estimators=300)

    # регрессия: classes=None, y — float
    X, y, feature_names, groups, _ = load_data(
        "data.csv", target_col="porosity", task="regression",
    )
    model = make_rf_reg(seed=42)
"""

from __future__ import annotations

import copy
import logging
import numbers
import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

# ---------------------------------------------------------------------------
# Логгер проекта
# ---------------------------------------------------------------------------
# Настройка формата/уровня — на стороне пользователя (basicConfig в скрипте
# запуска). Здесь только получаем логгер и гарантируем, что он не дублирует
# сообщения при повторном импорте модуля.
logger = logging.getLogger("otbor.common")
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Исключения проекта
# ---------------------------------------------------------------------------
class DataValidationError(ValueError):
    """Проблема с входными данными (отсутствует колонка, NaN,
    нечисловые значения и т.п.).

    Отдельный класс — чтобы в вызывающем коде можно было отличить
    «кривые данные» от прочих ValueError, возникающих внутри sklearn.
    """


class ConfigurationError(ValueError):
    """Некорректный параметр конфигурации (неизвестный тип CV,
    несовместимые опции и т.п.)."""


# ---------------------------------------------------------------------------
# Тип задачи
# ---------------------------------------------------------------------------
TASKS = ("classification", "regression")


def check_task(task):
    """Проверка типа задачи. Вызывается в load_data и конструкторах классов.

    Нужна, потому что хелперы ветвятся как «regression или иначе»:
    опечатка ('regresion') без проверки молча дала бы классификацию.

    Параметры
    ----------
    task : str
        'classification' или 'regression'.

    Возвращает
    -------
    task : str

    Исключения
    ----------
    ConfigurationError
        Неизвестный тип задачи.
    """
    if task not in TASKS:
        raise ConfigurationError(
            f"task должен быть одним из {TASKS}, получено {task!r}"
        )
    return task

# ===========================================================================
# 1. Фабрики моделей
# ===========================================================================
# Зачем фабрики вместо прямых вызовов конструкторов:
#   * Везде единообразно прокидывается seed.
#   * Линейные модели/KNN/SVC/MLP оборачиваются в Pipeline со
#     StandardScaler — чтобы не забыть масштабирование руками.
#   * Во всех Pipeline шаги называются одинаково: 'scaler' и 'model' —
#     общий код берёт модель как последний шаг pipe[-1].
#   * В evaluator'ах фабрика вызывается МНОГО раз (по разу на фолд/прогон),
#     поэтому важно, что make_*(seed) возвращает СВЕЖУЮ модель.
#
# Про воспроизводимость: при CV seed модели НЕ меняется автоматически —
# каждый фолд обучается с тем seed, что был передан в фабрику. Если методу
# нужны разные seed (например, бутстрэп-повторы), он сам передаёт
# seed=base_seed + run_index.


def make_rf(seed, n_estimators=500, max_depth=None, class_weight='balanced',
            **kwargs):
    """Random Forest Classifier.

    Дефолты: много деревьев (500), глубина не ограничена —
    «среднестатистический» RF.

    Параметры
    ----------
    seed : int
        random_state модели (бутстрэп и выбор признаков в узлах).
    n_estimators : int, default 500
        Число деревьев.
    max_depth : int | None, default None
        Максимальная глубина дерева; None — без ограничения.
    class_weight : str | dict | None, default 'balanced'
        Веса классов; 'balanced' — обратно пропорционально частоте класса.
    **kwargs
        Остальные параметры RandomForestClassifier
        (например, min_samples_leaf); не должны дублировать явные аргументы.

    Возвращает
    -------
    model : RandomForestClassifier
        Необученная модель.
    """
    # kwargs.setdefault("n_jobs", -1)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        class_weight=class_weight,
        **kwargs,
    )


def make_et(seed, n_estimators=500, max_depth=None, class_weight='balanced',
            **kwargs):
    """Extra Trees Classifier.

    Отличается от RF тем, что сплиты выбираются случайно — обычно
    чуть слабее по качеству, но заметно быстрее; полезен как
    быстрый «референс» при отладке пайплайна.

    Параметры
    ----------
    seed : int
        random_state модели (случайные пороги сплитов и выбор признаков).
    n_estimators : int, default 500
        Число деревьев.
    max_depth : int | None, default None
        Максимальная глубина дерева; None — без ограничения.
    class_weight : str | dict | None, default 'balanced'
        Веса классов; 'balanced' — обратно пропорционально частоте класса.
    **kwargs
        Остальные параметры ExtraTreesClassifier; не должны дублировать
        явные аргументы.

    Возвращает
    -------
    model : ExtraTreesClassifier
        Необученная модель.
    """
    # kwargs.setdefault("n_jobs", -1)
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        class_weight=class_weight,
        **kwargs,
    )


def make_xgb(seed, n_estimators=300, max_depth=6, learning_rate=0.1,
             subsample=0.8, colsample_bytree=0.8, importance_type='gain',
             **kwargs):
    """XGBoost Classifier.

    random_state влияет на subsample/colsample, поэтому при seed=None
    качество будет «плыть» — всегда передаём seed.

    class_weight у XGBClassifier нет (баланс классов задаётся только через
    sample_weight при fit), поэтому фабрика его не принимает: переданный
    через kwargs, он был бы молча проигнорирован.

    Параметры
    ----------
    seed : int
        random_state модели.
    n_estimators : int, default 300
        Число деревьев.
    max_depth : int, default 6
        Максимальная глубина дерева.
    learning_rate : float, default 0.1
        Шаг обучения.
    subsample : float, default 0.8
        Доля объектов для обучения каждого дерева.
    colsample_bytree : float, default 0.8
        Доля признаков для каждого дерева.
    importance_type : str, default 'gain'
        Тип встроенной важности (feature_importances_); 'gain' — как у
        make_lgbm, чтобы важности были сопоставимы.
    **kwargs
        Остальные параметры XGBClassifier; не должны дублировать явные
        аргументы.

    Возвращает
    -------
    model : XGBClassifier
        Необученная модель. Метки y должны быть целыми 0..K-1
        (кодируются в load_data); на fit каждый класс должен
        присутствовать в обучающей выборке.

    Исключения
    ----------
    ImportError
        xgboost не установлен.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "xgboost не установлен: pip install xgboost"
        ) from e
    # kwargs.setdefault("n_jobs", -1)
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        importance_type=importance_type,
        **kwargs,
    )


def make_lgbm(seed, n_estimators=300, max_depth=6, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
              verbose=-1, importance_type="gain", class_weight='balanced',
              **kwargs):
    """LightGBM Classifier.

    subsample в LightGBM работает только при subsample_freq > 0 (при 0
    бэггинг выключен и subsample молча игнорируется), поэтому по умолчанию
    subsample_freq=1 — чтобы заявленный subsample действительно применялся,
    как в make_xgb.

    verbose=-1 — глушим INFO/WARNING от нативной части LightGBM,
    иначе при сотнях вызовов фабрики логи забиты на 99%.

    importance_type="gain" — важность по суммарному приросту качества,
    обычно стабильнее дефолтного 'split' (кол-во сплитов, смещён
    в сторону высококардинальных признаков).

    Параметры
    ----------
    seed : int
        random_state модели.
    n_estimators : int, default 300
        Число деревьев.
    max_depth : int, default 6
        Максимальная глубина дерева.
    learning_rate : float, default 0.1
        Шаг обучения.
    subsample : float, default 0.8
        Доля объектов для каждого дерева (действует при subsample_freq > 0).
    colsample_bytree : float, default 0.8
        Доля признаков для каждого дерева.
    subsample_freq : int, default 1
        Как часто (в итерациях) выполнять бэггинг; 0 — бэггинг выключен.
    verbose : int, default -1
        Уровень логирования LightGBM; -1 — без вывода.
    importance_type : str, default 'gain'
        Тип встроенной важности (feature_importances_).
    class_weight : str | dict | None, default 'balanced'
        Веса классов; 'balanced' — обратно пропорционально частоте класса.
    **kwargs
        Остальные параметры LGBMClassifier; не должны дублировать явные
        аргументы.

    Возвращает
    -------
    model : LGBMClassifier
        Необученная модель.

    Исключения
    ----------
    ImportError
        lightgbm не установлен.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise ImportError(
            "lightgbm не установлен: pip install lightgbm"
        ) from e
    # kwargs.setdefault("n_jobs", -1)
    return LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        subsample_freq=subsample_freq,
        verbose=verbose,
        random_state=seed,
        importance_type=importance_type,
        class_weight=class_weight,
        **kwargs,
    )


def make_knn(seed=None, n_neighbors=5, weights="distance",
             metric="euclidean", **kwargs):
    """KNN Classifier, обёрнутый в Pipeline со StandardScaler.

    KNN чувствителен к масштабу признаков, поэтому scaler вшит сюда,
    а не вспоминается отдельно. Внутри Pipeline масштабирование
    обучается только на train-части фолда.

    Параметры
    ----------
    seed : int | None, default None
        Не используется (KNN детерминирован); оставлен для единообразия
        интерфейса model_factory(seed).
    n_neighbors : int, default 5
        Число соседей.
    weights : str, default 'distance'
        'uniform' или 'distance' (вес соседа обратно пропорционален
        расстоянию).
    metric : str, default 'euclidean'
        Метрика расстояния.
    **kwargs
        Остальные параметры KNeighborsClassifier; не должны дублировать
        явные аргументы.

    Возвращает
    -------
    model : Pipeline
        Необученный Pipeline из шагов 'scaler' (StandardScaler) и
        'model' (KNeighborsClassifier).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            **kwargs,
        )),
    ])


def make_logreg(seed, C=1.0, l1_ratio=0.0, class_weight="balanced",
                max_iter=2000, solver=None, **kwargs):
    """Logistic Regression, обёрнутая в Pipeline со StandardScaler.

    Тип регуляризации задаётся только через l1_ratio:
        l1_ratio = 0.0        -> L2          (solver="lbfgs", быстрый)
        l1_ratio = 1.0        -> L1          (solver="saga", единственный с L1)
        0.0 < l1_ratio < 1.0  -> elastic-net (solver="saga")
    solver=None -> выбирается автоматически по правилу выше.

    Параметр penalty не используется: в sklearn >= 1.8 он устарел.

    saga медленнее и капризнее к масштабированию — если видишь
    ConvergenceWarning, первым делом поднимай max_iter.

    Параметры
    ----------
    seed : int
        random_state модели (используется солвером saga).
    C : float, default 1.0
        Обратная сила регуляризации (меньше — сильнее регуляризация).
    l1_ratio : float, default 0.0
        Доля L1 в регуляризации, от 0.0 (чистый L2) до 1.0 (чистый L1).
    class_weight : str | dict | None, default 'balanced'
        Веса классов; 'balanced' — обратно пропорционально частоте класса.
    max_iter : int, default 2000
        Максимум итераций солвера.
    solver : str | None, default None
        None -> 'lbfgs' при l1_ratio=0.0, иначе 'saga'. Явно заданный
        солвер должен поддерживать выбранный l1_ratio.
    **kwargs
        Остальные параметры LogisticRegression; не должны дублировать
        явные аргументы.

    Возвращает
    -------
    model : Pipeline
        Необученный Pipeline из шагов 'scaler' (StandardScaler) и
        'model' (LogisticRegression).
    """
    if solver is None:
        solver = "lbfgs" if l1_ratio == 0.0 else "saga"

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            solver=solver,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=seed,
            **kwargs,
        )),
    ])


def make_svc(seed, C=1.0, kernel="rbf", gamma="scale",
             class_weight="balanced",
             calibration_method='sigmoid', calibration_cv=5, **kwargs):
    """SVC, обёрнутый в Pipeline(scaler -> SVC) + калибровка вероятностей.

    Почему CalibratedClassifierCV: SVC с probability=False не умеет
    predict_proba, а метрики roc_auc / logloss требуют вероятности.
    Калибровка (isotonic или sigmoid) обучает отдельную маленькую
    модель поверх decision_function, что надёжнее встроенного
    probability=True SVC.

    ВНИМАНИЕ: calibration_cv=5 (по умолчанию) означает, что на каждый
    фолд внешней CV приходится 5 дополнительных обучений SVC. На больших
    датасетах может быть дорого — тогда уменьшай calibration_cv,
    n_splits внешней CV или ядро SVC.

    Параметры
    ----------
    seed : int
        Не влияет на результат: SVC использует random_state только при
        probability=True, а вероятности здесь даёт калибратор. Оставлен
        для единообразия интерфейса model_factory(seed).
    C : float, default 1.0
        Параметр регуляризации SVC.
    kernel : str, default 'rbf'
        Тип ядра.
    gamma : str | float, default 'scale'
        Коэффициент ядра.
    class_weight : str | dict | None, default 'balanced'
        Веса классов; 'balanced' — обратно пропорционально частоте класса.
    calibration_method : str, default 'sigmoid'
        Метод калибровки: 'sigmoid' или 'isotonic'.
    calibration_cv : int, default 5
        Число внутренних фолдов калибровки.
    **kwargs
        Остальные параметры SVC (не калибратора); не должны дублировать
        явные аргументы.

    Возвращает
    -------
    model : CalibratedClassifierCV
        Необученный калибратор над Pipeline из шагов 'scaler'
        (StandardScaler) и 'model' (SVC).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            random_state=seed,
            **kwargs,
        )),
    ])
    return CalibratedClassifierCV(estimator=pipe,
                                  cv=calibration_cv,
                                  method=calibration_method)


def make_mlp(seed, hidden_layer_sizes=(64, 32), alpha=0.01,
             max_iter=2000, **kwargs):
    """MLP (нейросеть), обёрнутый в Pipeline со StandardScaler.

    Дефолтная сеть маленькая — это намеренно: задача отбора признаков,
    а не SOTA-качество. (128, 64, 32) — разумный следующий шаг.

    early_stopping=True включён по умолчанию (через kwargs): 10% train-части
    уходит на внутреннюю стратифицированную валидацию. При малом числе
    объектов редкого класса это может падать — отключается
    early_stopping=False. Параметра class_weight у модели нет.

    Параметры
    ----------
    seed : int
        random_state модели: инициализация весов, перемешивание и разбиение
        для early stopping. Результат заметно зависит от seed.
    hidden_layer_sizes : tuple, default (64, 32)
        Число нейронов в каждом скрытом слое.
    alpha : float, default 0.01
        Сила L2-регуляризации.
    max_iter : int, default 2000
        Максимум итераций обучения.
    **kwargs
        Остальные параметры MLPClassifier; не должны дублировать явные
        аргументы.

    Возвращает
    -------
    model : Pipeline
        Необученный Pipeline из шагов 'scaler' (StandardScaler) и
        'model' (MLPClassifier).
    """
    kwargs.setdefault("early_stopping", True)   # страховка от перебора
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            random_state=seed,
            **kwargs,
        )),
    ])


# ---------------------------------------------------------------------------
# Фабрики регрессии
# ---------------------------------------------------------------------------
# Те же соглашения, что у фабрик классификации (seed, Pipeline со шагами
# 'scaler' и 'model', **kwargs). Отличия — только в отсутствии
# class_weight и калибровки вероятностей.


def make_rf_reg(seed, n_estimators=500, max_depth=None, **kwargs):
    """Random Forest Regressor.

    В отличие от классификатора, по умолчанию в sklearn
    max_features=1.0: в каждом узле перебираются все признаки, деревья
    отличаются только бутстрэпом. Это влияет на MDI: сильный признак
    чаще забирает сплиты у коррелированных с ним. Для случайного
    подмножества признаков в узле — max_features='sqrt' или доля (0.33)
    через kwargs.

    Параметры
    ----------
    seed : int
        random_state модели (бутстрэп и выбор признаков в узлах).
    n_estimators : int, default 500
        Число деревьев.
    max_depth : int | None, default None
        Максимальная глубина дерева; None — без ограничения.
    **kwargs
        Остальные параметры RandomForestRegressor (например,
        min_samples_leaf, max_features); не должны дублировать явные
        аргументы.

    Возвращает
    -------
    model : RandomForestRegressor
        Необученная модель.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        **kwargs,
    )


def make_et_reg(seed, n_estimators=500, max_depth=None, **kwargs):
    """Extra Trees Regressor.

    Как и у make_rf_reg, по умолчанию max_features=1.0; случайность —
    в порогах сплитов.

    Параметры
    ----------
    seed : int
        random_state модели (случайные пороги сплитов).
    n_estimators : int, default 500
        Число деревьев.
    max_depth : int | None, default None
        Максимальная глубина дерева; None — без ограничения.
    **kwargs
        Остальные параметры ExtraTreesRegressor; не должны дублировать
        явные аргументы.

    Возвращает
    -------
    model : ExtraTreesRegressor
        Необученная модель.
    """
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        **kwargs,
    )


def make_xgb_reg(seed, n_estimators=300, max_depth=6, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, importance_type='gain',
                 **kwargs):
    """XGBoost Regressor (по умолчанию objective='reg:squarederror').

    Параметры
    ----------
    seed : int
        random_state модели.
    n_estimators : int, default 300
        Число деревьев.
    max_depth : int, default 6
        Максимальная глубина дерева. На малых выборках (сотни объектов)
        стоит уменьшать (3–4): иначе деревья подгоняются под шум.
    learning_rate : float, default 0.1
        Шаг обучения.
    subsample : float, default 0.8
        Доля объектов для обучения каждого дерева.
    colsample_bytree : float, default 0.8
        Доля признаков для каждого дерева.
    importance_type : str, default 'gain'
        Тип встроенной важности (feature_importances_), как у make_xgb.
    **kwargs
        Остальные параметры XGBRegressor; не должны дублировать явные
        аргументы.

    Возвращает
    -------
    model : XGBRegressor
        Необученная модель.

    Исключения
    ----------
    ImportError
        xgboost не установлен.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError(
            "xgboost не установлен: pip install xgboost"
        ) from e
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        importance_type=importance_type,
        **kwargs,
    )


def make_lgbm_reg(seed, n_estimators=300, max_depth=6, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                  verbose=-1, importance_type="gain", **kwargs):
    """LightGBM Regressor.

    subsample_freq=1, verbose=-1, importance_type='gain' — по тем же
    причинам, что в make_lgbm.

    На малых выборках важен min_child_samples (по умолчанию 20 в
    LightGBM): при ~200 объектах в train это ограничивает глубину сильнее,
    чем max_depth. Задаётся через kwargs.

    Параметры
    ----------
    seed : int
        random_state модели.
    n_estimators : int, default 300
        Число деревьев.
    max_depth : int, default 6
        Максимальная глубина дерева.
    learning_rate : float, default 0.1
        Шаг обучения.
    subsample : float, default 0.8
        Доля объектов для каждого дерева (действует при subsample_freq > 0).
    colsample_bytree : float, default 0.8
        Доля признаков для каждого дерева.
    subsample_freq : int, default 1
        Как часто (в итерациях) выполнять бэггинг; 0 — бэггинг выключен.
    verbose : int, default -1
        Уровень логирования LightGBM; -1 — без вывода.
    importance_type : str, default 'gain'
        Тип встроенной важности (feature_importances_).
    **kwargs
        Остальные параметры LGBMRegressor; не должны дублировать явные
        аргументы.

    Возвращает
    -------
    model : LGBMRegressor
        Необученная модель.

    Исключения
    ----------
    ImportError
        lightgbm не установлен.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise ImportError(
            "lightgbm не установлен: pip install lightgbm"
        ) from e
    return LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        subsample_freq=subsample_freq,
        verbose=verbose,
        random_state=seed,
        importance_type=importance_type,
        **kwargs,
    )


def make_knn_reg(seed=None, n_neighbors=5, weights="distance",
                 metric="euclidean", **kwargs):
    """KNN Regressor, обёрнутый в Pipeline со StandardScaler.

    Параметры
    ----------
    seed : int | None, default None
        Не используется (KNN детерминирован); оставлен для единообразия
        интерфейса model_factory(seed).
    n_neighbors : int, default 5
        Число соседей.
    weights : str, default 'distance'
        'uniform' или 'distance'.
    metric : str, default 'euclidean'
        Метрика расстояния.
    **kwargs
        Остальные параметры KNeighborsRegressor; не должны дублировать
        явные аргументы.

    Возвращает
    -------
    model : Pipeline
        Необученный Pipeline из шагов 'scaler' (StandardScaler) и
        'model' (KNeighborsRegressor).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            **kwargs,
        )),
    ])


def make_linreg(seed, alpha=1.0, l1_ratio=0.0, max_iter=10000, **kwargs):
    """Регуляризованная линейная регрессия в Pipeline со StandardScaler.
    Аналог make_logreg для регрессии.

    Тип регуляризации задаётся через l1_ratio:
        l1_ratio = 0.0        -> L2 (Ridge, точное решение)
        l1_ratio = 1.0        -> L1 (ElasticNet с l1_ratio=1, т.е. Lasso)
        0.0 < l1_ratio < 1.0  -> elastic-net (ElasticNet)
    Ridge при l1_ratio=0, а не ElasticNet: координатный спуск ElasticNet
    при чистом L2 сходится плохо (sklearn об этом предупреждает).

    ВНИМАНИЕ: масштаб alpha у Ridge и ElasticNet разный. ElasticNet
    минимизирует 1/(2n)·||y - Xw||² + alpha·(...), Ridge — ||y - Xw||² +
    alpha·||w||², т.е. без деления на n. Одно и то же alpha при l1_ratio=0
    и l1_ratio>0 даёт несопоставимую силу регуляризации.

    Параметры
    ----------
    seed : int
        random_state модели (используется ElasticNet при
        selection='random' и Ridge с солверами sag/saga).
    alpha : float, default 1.0
        Сила регуляризации (больше — сильнее; обратна C в make_logreg).
    l1_ratio : float, default 0.0
        Доля L1 в регуляризации, от 0.0 (чистый L2) до 1.0 (чистый L1).
    max_iter : int, default 10000
        Максимум итераций координатного спуска (ElasticNet).
    **kwargs
        Остальные параметры Ridge / ElasticNet; не должны дублировать
        явные аргументы.

    Возвращает
    -------
    model : Pipeline
        Необученный Pipeline из шагов 'scaler' (StandardScaler) и
        'model' (Ridge или ElasticNet). coef_ модели — одномерный (p,).
    """
    if l1_ratio == 0.0:
        model = Ridge(alpha=alpha, random_state=seed, **kwargs)
    else:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                           max_iter=max_iter, random_state=seed, **kwargs)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def make_svr(seed=None, C=1.0, epsilon=0.1, kernel="rbf", gamma="scale",
             **kwargs):
    """SVR, обёрнутый в Pipeline со StandardScaler.

    Калибровка, как в make_svc, не нужна: регрессия не требует
    вероятностей.

    epsilon задаётся в единицах y: при большом масштабе целевой
    переменной дефолт 0.1 почти ничего не значит, при малом — может
    занять весь диапазон. y не масштабируется, поэтому epsilon и C
    стоит подбирать под данные.

    Параметры
    ----------
    seed : int | None, default None
        Не используется (SVR детерминирован); оставлен для единообразия
        интерфейса model_factory(seed).
    C : float, default 1.0
        Параметр регуляризации.
    epsilon : float, default 0.1
        Ширина трубки, внутри которой ошибка не штрафуется (в единицах y).
    kernel : str, default 'rbf'
        Тип ядра.
    gamma : str | float, default 'scale'
        Коэффициент ядра.
    **kwargs
        Остальные параметры SVR; не должны дублировать явные аргументы.

    Возвращает
    -------
    model : Pipeline
        Необученный Pipeline из шагов 'scaler' (StandardScaler) и
        'model' (SVR).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            C=C,
            epsilon=epsilon,
            kernel=kernel,
            gamma=gamma,
            **kwargs,
        )),
    ])


def make_mlp_reg(seed, hidden_layer_sizes=(64, 32), alpha=0.01,
                 max_iter=2000, **kwargs):
    """MLP Regressor, обёрнутый в Pipeline со StandardScaler.

    early_stopping=True по умолчанию (через kwargs), как в make_mlp:
    10% train-части уходит на внутреннюю валидацию. При сотнях объектов
    это 20–30 точек — остановка становится шумной; тогда
    early_stopping=False и меньший max_iter.

    y не масштабируется: при большом масштабе целевой переменной MLP
    обучается плохо. Если это мешает — масштабировать y до load_data или
    обернуть модель в TransformedTargetRegressor.

    Параметры
    ----------
    seed : int
        random_state модели: инициализация весов, перемешивание и разбиение
        для early stopping.
    hidden_layer_sizes : tuple, default (64, 32)
        Число нейронов в каждом скрытом слое.
    alpha : float, default 0.01
        Сила L2-регуляризации.
    max_iter : int, default 2000
        Максимум итераций обучения.
    **kwargs
        Остальные параметры MLPRegressor; не должны дублировать явные
        аргументы.

    Возвращает
    -------
    model : Pipeline
        Необученный Pipeline из шагов 'scaler' (StandardScaler) и
        'model' (MLPRegressor).
    """
    kwargs.setdefault("early_stopping", True)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            max_iter=max_iter,
            random_state=seed,
            **kwargs,
        )),
    ])


# ===========================================================================
# 2. Кросс-валидация
# ===========================================================================

def _is_int_seed(value):
    """int / np.integer, но не bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def check_cv(cv, random_state=None, task="classification"):
    """Проверка и нормализация CV-сплиттера. Вызывается в конструкторе
    каждого класса (ImportanceEvaluator, FeatureSelector, ModelValidator).

    Параметры
    ----------
    cv : sklearn CV splitter | None
        None -> дефолтный сплиттер с n_splits=5, shuffle=True,
        random_state=random_state: StratifiedKFold для классификации,
        KFold для регрессии (стратификация по непрерывному y в sklearn
        не поддерживается).
    random_state : int | None
        Используется только для дефолтного сплиттера (cv=None). У
        переданного сплиттера действует его собственный random_state.
    task : {'classification', 'regression'}, default 'classification'
        Тип задачи; влияет только на дефолтный сплиттер.

    Проверки (нарушение любой -> ConfigurationError):
        * у cv есть метод split;
        * random_state сплиттера — int или None (RandomState-объекты не
          поддерживаются: cv_for_run сдвигает seed на run_index);
        * shuffle=True требует заданного random_state — иначе каждый вызов
          split() даёт новые фолды, и «фиксированный» self.cv перестаёт
          быть фиксированным;
        * shuffle=False с заданным random_state — бессмысленно;
        * всегда случайные сплиттеры (ShuffleSplit, GroupShuffleSplit)
          требуют заданного random_state.
    Детерминированные сплиттеры без random_state (LeaveOneGroupOut)
    проходят без изменений.

    Возвращает
    -------
    cv : sklearn CV splitter
    """
    if cv is None:
        if random_state is None:
            raise ConfigurationError(
                "cv=None: для дефолтного сплиттера (shuffle=True) нужен "
                "random_state, иначе фолды не воспроизводимы"
            )
        splitter = KFold if task == "regression" else StratifiedKFold
        cv = splitter(n_splits=5, shuffle=True, random_state=random_state)

    if not hasattr(cv, "split"):
        raise ConfigurationError(
            f"cv должен быть sklearn-сплиттером (объект с методом split), "
            f"получено: {type(cv).__name__}"
        )

    name = type(cv).__name__
    seed = getattr(cv, "random_state", None)
    shuffle = getattr(cv, "shuffle", None)

    if seed is not None and not _is_int_seed(seed):
        raise ConfigurationError(
            f"{name}: random_state должен быть int или None, "
            f"получено {type(seed).__name__}"
        )
    if shuffle is not None and bool(shuffle) and seed is None:
        raise ConfigurationError(
            f"{name}(shuffle=True) без random_state: фолды не воспроизводимы"
        )
    if shuffle is not None and not bool(shuffle) and seed is not None:
        raise ConfigurationError(
            f"{name}(shuffle=False) с random_state={seed}: random_state "
            "не имеет эффекта"
        )
    if shuffle is None and hasattr(cv, "random_state") and seed is None:
        raise ConfigurationError(
            f"{name} всегда случайный: задай random_state"
        )
    return cv


def make_folds(cv, X, y, groups=None, check_classes=True,
               check_val_classes=False, task="classification"):
    """Фолды cv как список пар (train_indices, val_indices).

    Единая точка, через которую groups попадают в cv.split: GroupKFold и
    LeaveOneGroupOut их требуют, StratifiedKFold игнорирует. Возвращается
    список, а не генератор: его можно переиспользовать и передавать
    в sklearn как cv=.

    Параметры
    ----------
    cv : sklearn CV splitter
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    groups : array-like, shape (n_samples,) | None
    check_classes : bool, default True
        Проверять, что в каждом train-фолде есть все классы y. Модель не
        может предсказать класс, которого не видела, а predict_proba теряет
        столбец — метрики по фолдам перестают быть сопоставимыми
        (XGBoost в этом случае падает на fit). Отключать осознанно.
    check_val_classes : bool, default False
        То же для val-фолдов. Нужно методам, которые считают на val-части
        метрику, требующую все классы (roc_auc_ovr в permutation importance).
        По умолчанию выключено: при LeaveOneGroupOut val — одна скважина,
        и в ней часто представлены не все классы.
    task : {'classification', 'regression'}, default 'classification'
        При регрессии классов нет: check_classes и check_val_classes
        игнорируются (поэтому методы классов могут держать
        check_val_classes=True по умолчанию для обеих задач).

    Возвращает
    -------
    splits : list[tuple[np.ndarray, np.ndarray]]

    Исключения
    ----------
    ConfigurationError
        Сплиттеру нужны groups, а они не заданы; либо (при check_classes /
        check_val_classes) в каком-то train / val-фолде отсутствуют классы.
    """
    y = np.asarray(y)
    try:
        splits = list(cv.split(X, y, groups))
    except ValueError as e:
        if groups is None and "groups" in str(e).lower():
            raise ConfigurationError(
                f"{type(cv).__name__} требует groups (group_col в load_data)"
            ) from e
        raise

    if task == "regression":
        return splits

    parts = []
    if check_classes:
        parts.append(("train", 0, "check_classes"))
    if check_val_classes:
        parts.append(("val", 1, "check_val_classes"))

    all_classes = np.unique(y)
    for i, split in enumerate(splits):
        for part, pos, flag in parts:
            missing = np.setdiff1d(all_classes, np.unique(y[split[pos]]))
            if missing.size:
                raise ConfigurationError(
                    f"Фолд {i}: в {part} отсутствуют классы "
                    f"{missing.tolist()}. Измени разбиение (число фолдов, "
                    f"объединение классов) или отключи {flag}."
                )
    return splits


def cv_for_run(cv, run_index):
    """Клон сплиттера cv для run_index-го прогона со сдвинутым random_state.

    Зачем: вызов .split() у одного и того же сплиттера с фиксированным
    random_state возвращает идентичные фолды. Без сдвига seed повторные
    прогоны (n_runs) усредняли бы только шум модели, а не шум разбиения.

    Параметры
    ----------
    cv : sklearn CV splitter
        Например StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
        Предварительно проверен check_cv.
    run_index : int
        Номер прогона (0-based).

    Возвращает
    -------
    cv_run : sklearn CV splitter
        random_state = cv.random_state + run_index, если random_state задан.
        Если атрибута нет или он None (LeaveOneGroupOut, shuffle=False) —
        возвращается копия без изменений: разбиение во всех прогонах
        одинаково, прогоны отличаются только seed моделей.
    """
    cv_run = copy.deepcopy(cv)
    seed = getattr(cv, "random_state", None)
    if seed is not None:
        if not _is_int_seed(seed):
            raise ConfigurationError(
                f"{type(cv).__name__}: random_state должен быть int, "
                f"получено {type(seed).__name__}"
            )
        cv_run.random_state = int(seed) + run_index
    return cv_run


# ===========================================================================
# 3. Хелперы методов важности и метрик
# ===========================================================================
# Единственное место, где по типу задачи выбираются функция sklearn для
# filter-методов, выход модели для SHAP и дефолтная метрика. Методы классов
# передают сюда self.task и сами по задаче не ветвятся.

_FILTER_FUNCS = {
    "classification": {
        "mi": mutual_info_classif,
        "anova_f": f_classif,
    },
    "regression": {
        "mi": mutual_info_regression,
        "anova_f": f_regression,
    },
}

_DEFAULT_SCORING = {
    "classification": "f1_macro",
    "regression": "r2",
}

# Permutation importance: падение метрики при перестановке. Дискретная
# f1_macro для этого грубая (меняется скачками), поэтому для классификации
# — непрерывная roc_auc_ovr.
_DEFAULT_PI_SCORING = {
    "classification": "roc_auc_ovr",
    "regression": "r2",
}


def get_filter_func(name, task="classification"):
    """Функция sklearn для filter-метода по его имени и типу задачи.

    Параметры
    ----------
    name : str
        'mi' -> mutual_info_classif / mutual_info_regression;
        'anova_f' -> f_classif / f_regression. Для регрессии это F-тест
        значимости линейной связи признака с y (эквивалент квадрата
        корреляции Пирсона), а не ANOVA по классам; имя общее, чтобы
        у compute_anova_f был один label для обеих задач.
    task : {'classification', 'regression'}, default 'classification'

    Возвращает
    -------
    func : callable
        Функция с интерфейсом func(X, y, **kwargs).

    Исключения
    ----------
    ConfigurationError
        Неизвестное имя метода.
    """
    funcs = _FILTER_FUNCS[task]
    if name not in funcs:
        raise ConfigurationError(
            f"Неизвестный filter-метод {name!r}. "
            f"Доступны: {list(funcs)}"
        )
    return funcs[name]


def get_predict_fn(model, task="classification"):
    """Выход модели, который объясняют модельно-независимые методы (SHAP).

    Классификация -> predict_proba (вероятности классов);
    регрессия -> predict.
    """
    return model.predict if task == "regression" else model.predict_proba


def get_predict_output(task="classification"):
    """Что возвращает get_predict_fn в терминах model_output у SHAP.

    Классификация -> 'probability' (predict_proba); регрессия -> 'raw'
    (predict — и есть «сырой» выход модели). compute_shap по этому
    значению выбирает дефолтный model_output и проверяет объяснители,
    не ветвясь по задаче.
    """
    return "raw" if task == "regression" else "probability"


def check_mi_neighbors(y, n_neighbors, task="classification"):
    """Проверка n_neighbors для mutual information на train-части фолда.

    Классификация: sklearn при n_neighbors >= размера класса молча
    уменьшает k для этого класса, и оценки по классам становятся
    несопоставимыми — поэтому ошибка, а не тихая подмена.
    Регрессия: ограничения по классам нет, проверка не нужна.

    Исключения
    ----------
    ConfigurationError
        n_neighbors не меньше размера самого малого класса.
    """
    if task == "regression":
        return
    counts = np.bincount(np.asarray(y))
    n_min = int(counts[counts > 0].min())
    if n_neighbors >= n_min:
        raise ConfigurationError(
            f"MI: n_neighbors={n_neighbors} должно быть меньше размера "
            f"самого малого класса в train-фолде ({n_min})"
        )


def get_default_scoring(task="classification", method=None):
    """Метрика по умолчанию (строка sklearn-scorer'а) для типа задачи.

    Методы классов принимают scoring=None и подставляют её через этот
    хелпер.

    Параметры
    ----------
    task : {'classification', 'regression'}, default 'classification'
    method : {None, 'pi'}, default None
        None — метрика качества модели: 'f1_macro' / 'r2' (r2 безразмерна,
        сравнима между датасетами);
        'pi' — метрика permutation importance: 'roc_auc_ovr' / 'r2'.
    """
    if method == "pi":
        return _DEFAULT_PI_SCORING[task]
    return _DEFAULT_SCORING[task]


# ===========================================================================
# 4. Загрузка данных
# ===========================================================================

def load_data(filepath, target_col, cols_to_drop=None, group_col=None,
              task="classification"):
    """Загрузка CSV и подготовка массивов для отбора признаков.

    Проверки (нарушение любой -> DataValidationError):
        * target_col, group_col и все колонки cols_to_drop существуют;
        * все признаки числовые;
        * target_col и group_col не указаны в cols_to_drop и различаются;
        * нет NaN в X, y и groups, нет inf в X;
        * классификация: в y не меньше двух классов;
        * регрессия: y числовой, без inf.

    Классификация: метки перекодируются в целые 0..K-1 (порядок — по
    возрастанию исходных меток, np.unique). Соответствие «исходная метка
    -> код» пишется в лог и возвращается массивом classes.
    Регрессия: y приводится к float64 без кодирования, classes=None;
    в лог пишется сводка y (min, max, mean, std).

    Параметры
    ----------
    filepath : str
        Путь к CSV-файлу.
    target_col : str
        Имя колонки с целевой переменной. Из признаков исключается.
    cols_to_drop : list[str] | None, default None
        Колонки, удаляемые до формирования X (id, даты и т.п.).
        Каждая обязана существовать в файле.
    group_col : str | None, default None
        Колонка с группами (well_id). Из признаков исключается,
        возвращается отдельным массивом.
    task : {'classification', 'regression'}, default 'classification'
        Тип задачи; определяет обработку y.

    Возвращает
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype float64
    y : np.ndarray, shape (n_samples,)
        Классификация — int64, коды классов 0..K-1;
        регрессия — float64, исходные значения.
    feature_names : list[str]
        Имена колонок X в порядке следования.
    groups : np.ndarray, shape (n_samples,) | None
        None, если group_col не задан.
    classes : np.ndarray, shape (K,) | None
        classes[k] — исходная метка класса с кодом k; None для регрессии.
    """
    check_task(task)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Загружен {filepath}: {df.shape[0]} строк, {df.shape[1]} колонок")

    # --- целевая ---
    if target_col not in df.columns:
        raise DataValidationError(
            f"target_col={target_col!r} отсутствует. "
            f"Доступны: {list(df.columns)}"
        )
    y_raw = df[target_col].to_numpy()

    # --- колонки на удаление ---
    cols_to_drop = list(cols_to_drop) if cols_to_drop else []
    missing = [c for c in cols_to_drop if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"cols_to_drop содержит несуществующие колонки: {missing}"
        )
    if target_col in cols_to_drop:
        raise DataValidationError(
            f"target_col={target_col!r} указан в cols_to_drop"
        )
    if group_col is not None:
        if group_col == target_col:
            raise DataValidationError("group_col совпадает с target_col")
        if group_col in cols_to_drop:
            raise DataValidationError(
                f"group_col={group_col!r} указан в cols_to_drop"
            )
    df = df.drop(columns=cols_to_drop)

    # --- группы (до формирования X, чтобы не попали в признаки) ---
    groups = None
    if group_col is not None:
        if group_col not in df.columns:
            raise DataValidationError(f"group_col={group_col!r} отсутствует")
        groups = df[group_col].to_numpy()
        df = df.drop(columns=[group_col])
        if pd.isna(groups).any():
            raise DataValidationError(
                f"В group_col={group_col!r} есть NaN — группы должны "
                "быть заданы для всех объектов"
            )

    # --- признаки ---
    feature_names = list(df.columns)
    feature_names.remove(target_col)

    X_df = df[feature_names]
    non_numeric = [c for c in feature_names
                   if not pd.api.types.is_numeric_dtype(X_df[c])]
    if non_numeric:
        raise DataValidationError(
            f"Нечисловые признаки: {non_numeric}"
        )
    X = X_df.to_numpy(dtype=np.float64)

    # --- пропуски ---
    n_nan_x = int(np.isnan(X).sum())
    if n_nan_x:
        nan_cols = [c for c in feature_names if X_df[c].isna().any()]
        raise DataValidationError(
            f"NaN в признаках: {n_nan_x} значений, колонки: {nan_cols}"
        )
    if np.isinf(X).any():
        inf_cols = [c for c, bad in zip(feature_names, np.isinf(X).any(axis=0)) if bad]
        raise DataValidationError(f"inf в признаках, колонки: {inf_cols}")
    if pd.isna(y_raw).any():
        raise DataValidationError(f"NaN в целевой переменной {target_col!r}")

    groups_info = f", групп={len(np.unique(groups))}" if groups is not None else ""

    # --- регрессия: y как есть ---
    if task == "regression":
        if not pd.api.types.is_numeric_dtype(y_raw):
            raise DataValidationError(
                f"task='regression': целевая {target_col!r} нечисловая"
            )
        y = y_raw.astype(np.float64)
        if np.isinf(y).any():
            raise DataValidationError(
                f"inf в целевой переменной {target_col!r}"
            )
        logger.info(
            f"Целевая {target_col!r}: min={y.min():.4g}, max={y.max():.4g}, "
            f"mean={y.mean():.4g}, std={y.std():.4g}"
        )
        logger.info(
            f"Итог: n={X.shape[0]}, признаков={X.shape[1]}, "
            f"регрессия{groups_info}"
        )
        return X, y, feature_names, groups, None

    # --- кодирование классов ---
    classes, y = np.unique(y_raw, return_inverse=True)
    y = y.astype(np.int64)
    if len(classes) < 2:
        raise DataValidationError(
            f"В {target_col!r} меньше двух классов: {list(classes)}"
        )
    counts = np.bincount(y)
    mapping = ", ".join(
        f"{c!r} -> {k} (n={counts[k]})" for k, c in enumerate(classes)
    )
    logger.info(f"Кодирование классов (исходная метка -> код): {mapping}")

    logger.info(
        f"Итог: n={X.shape[0]}, признаков={X.shape[1]}, классов={len(classes)}{groups_info}"
    )
    return X, y, feature_names, groups, classes


def check_data(X, y, feature_names, groups=None):
    """Проверка согласованности массивов; вызывается в конструкторе
    каждого класса (ImportanceEvaluator, FeatureSelector, ModelValidator).

    Содержимое (NaN, нечисловые признаки) проверяет load_data; здесь —
    только формы, т.к. классы могут получать X, y не из load_data
    (например, части A/B после разбиения в research).

    Параметры
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    feature_names : list[str]
        len == n_features.
    groups : array-like, shape (n_samples,) | None, default None

    Возвращает
    -------
    X : np.ndarray
    y : np.ndarray
    feature_names : list[str]
    groups : np.ndarray | None

    Исключения
    ----------
    DataValidationError
        Несогласованные размеры X, y, groups, feature_names.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    feature_names = list(feature_names)

    if X.ndim != 2:
        raise DataValidationError(
            f"X должен быть двумерным, получено ndim={X.ndim}"
        )
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise DataValidationError(
            f"y должен иметь форму ({X.shape[0]},), получено {y.shape}"
        )
    if len(feature_names) != X.shape[1]:
        raise DataValidationError(
            f"feature_names: {len(feature_names)} имён на "
            f"{X.shape[1]} признаков"
        )
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise DataValidationError(
                f"groups должен иметь форму {y.shape}, "
                f"получено {groups.shape}"
            )
    return X, y, feature_names, groups