# optimal-set-of-attirbutes

Библиотека для отбора признаков (feature selection) в задачах классификации на табличных данных: ранжирование признаков разными методами, выбор оптимального числа признаков по кривой качества и статистическое сравнение методов.

Код не зависит от происхождения признаков и работает на любом табличном датасете. Первичная проверка ведётся на открытых данных. Целевая область — сейсмика и скважинные данные, где объекты сгруппированы по скважинам и кросс-валидация должна учитывать группы.

**Стадия:** прототип. Сейчас поддерживается только классификация; регрессия запланирована.

## Модули

| Модуль | Класс | Назначение | Статус |
|---|---|---|---|
| `common.py` | — | фабрики моделей, утилиты кросс-валидации, загрузка и проверка данных | готов |
| `importance.py` | `ImportanceEvaluator` | ранжирование признаков: MI, ANOVA F, MDI, L1, permutation importance, SHAP | готов |
| `validation.py` | `ModelValidator` | качество модели на топ-N признаков, выбор N и итогового набора | готов |
| `selection.py` | `FeatureSelector` | перебор признаков: forward, backward, RFE | в планах |
| `statistics.py` | `StatisticalAnalyzer` | Friedman, Nemenyi, стабильность (Jaccard) | в планах |

## Установка

Python 3.12. Зависимости:

```bash
pip install numpy pandas scikit-learn joblib shap
pip install xgboost lightgbm   # опционально, для make_xgb / make_lgbm
```

Проверено на numpy 2.5, pandas 3.0, scikit-learn 1.9, shap 0.54, xgboost 3.4, lightgbm 4.7.

## Пример

```python
from sklearn.model_selection import StratifiedKFold

from common import load_data, make_rf
from importance import ImportanceEvaluator
from validation import ModelValidator

# CSV: признаки + целевая колонка (+ опционально колонка групп).
X, y, feature_names, groups, classes = load_data(
    "data.csv", target_col="Y", group_col=None, cols_to_drop=["id"],
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. Ранжирования признаков, усреднённые по 5 прогонам CV.
ev = ImportanceEvaluator(X, y, feature_names, cv=cv, n_runs=5)
ev.compute_mi("mi")
ev.compute_pi(lambda s: make_rf(s, n_estimators=100), "pi_rf",
              scoring=["roc_auc_ovr"])

# 2. Кривая качества модели по числу признаков и выбор набора.
val = ModelValidator(X, y, feature_names, cv=cv)
val.evaluate_model_on_subsets(
    lambda s: make_rf(s, n_estimators=100), ev.get_ranking("mi"),
    "rf_mi", scoring=["f1_macro"],
)
print(val.get_validation_summary("rf_mi"))           # mean/std по фолдам для каждого N
subset = val.get_best_subset("rf_mi", "f1_macro")    # N по правилу 1-SE
```

Для данных, сгруппированных по скважинам, вместо `StratifiedKFold` передаётся `GroupKFold` или `LeaveOneGroupOut` и `groups` из `load_data(..., group_col="well_id")`.

## Принципы

- **Единый CV-сплиттер.** Сплиттер sklearn передаётся в конструктор каждого класса. Ранжирования усредняются по прогонам с разными разбиениями, а кривые качества разных методов строятся на одних и тех же фолдах — оценки парные и сравнимы статистическими тестами.
- **Воспроизводимость.** Модели создаются фабриками вида `model_factory(seed)`, seed всегда задаётся явно; сплиттеры со случайным разбиением обязаны иметь `random_state`.
- **Выбор числа признаков.** По умолчанию — правило одной стандартной ошибки (1-SE): наименьшее N, качество которого отличается от лучшего меньше чем на стандартную ошибку по фолдам. Доступно и правило максимума (`rule="max"`).
- **Библиотека не делит данные сама.** Если ранжирование и выбор N делаются на одних и тех же данных, оценка качества оптимистична. Честная схема — отдельные части данных для ранжирования, выбора набора и финальной проверки — реализуется вызывающим кодом.

## Прочее

`old/attributes_splited.py` — файл из другого проекта, к библиотеке не относится.
