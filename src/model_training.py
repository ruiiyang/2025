import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv

def plot_feature_importance(feature_names, importance_values, title):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance_values)
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.tight_layout()
    return plt

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names=None):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'AUC-ROC': roc_auc_score(y_test, y_test_prob),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

    if feature_names is not None:
        if hasattr(model, 'feature_importances_'):
            #random forest
            values = model.feature_importances_
            title = f"{type(model).__name__} Feature Importance"
            plt = plot_feature_importance(feature_names, values, title)
            plt.show()
        elif hasattr(model, 'coef_'):
            # logistic regression
            values = np.abs(model.coef_[0])
            title = f"{type(model).__name__} Feature Coefficients (Absolute)"
            plt = plot_feature_importance(feature_names, values, title)
            plt.show()

    return metrics

def train_models(X_train, X_test, y_train, y_test, feature_names=None, random_state=5117):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hparam_search_{timestamp}.csv")

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'params', 'mean_test_score', 'std_test_score', 'rank_test_score'])

    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.7]
        },
       'Logistic Regression': [
            {
                'penalty': ['l2'],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'sag', 'newton-cg', 'saga']
            },
            {
                'penalty': ['l1'],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga']
            },
            {
                'penalty': [None],
                'solver': ['lbfgs', 'sag', 'newton-cg', 'saga']
            },
            {
                'penalty': ['elasticnet'],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['saga'],
                'l1_ratio': [0.3, 0.5, 0.7]
            }
       ]
    }

    base_models = {
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000)
    }

    results = {}
    for name in base_models.keys():
        grid_search = GridSearchCV(
            estimator=base_models[name],
            param_grid=param_grids[name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for params, mean_score, std_score, rank in zip(
                grid_search.cv_results_['params'],
                grid_search.cv_results_['mean_test_score'],
                grid_search.cv_results_['std_test_score'],
                grid_search.cv_results_['rank_test_score']
            ):
                writer.writerow([
                    name,
                    str(params),
                    f"{mean_score:.4f}",
                    f"{std_score:.4f}",
                    rank
                ])

        best_model = grid_search.best_estimator_
        metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test, feature_names)
        metrics['Best Params'] = grid_search.best_params_
    
        results[name] = metrics
        print(f"{name} Best parameters: {grid_search.best_params_}")

    return results

