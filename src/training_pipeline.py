import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import joblib

def train_model(X_train, y_train, n_iter=20, cv=5):

    # best params = {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.01, 'colsample_bytree': 1}
    # model = XGBClassifier(
    #     n_estimators=100,
    #     max_depth=7,
    #     learning_rate=0.01,
    #     subsample=0.8,
    #     colsample_bytree=1,
    #     random_state=42)


    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'min_child_weight': [1, 3, 5]
    }

    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Guardar el mejor modelo
    joblib.dump(best_model, "models/churn_model.pkl")

    print("Mejores hiperparámetros:", search.best_params_)
    print("Mejor F1 en CV:", search.best_score_)

    return best_model
