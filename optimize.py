import mlflow
import optuna
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('water_potability.csv')

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def optimize_model(trial):
  params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': trial.suggest_loguniform('eta', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 0.01, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'use_label_encoder': False
    }
  model = XGBClassifier(**params)

  with mlflow.start_run(run_name=f"XGBoost con lr {params['eta']}"):
        mlflow.log_params(params)
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        mlflow.log_metric("valid_mae", mae)
        
        return mae
  
def main():
    mlflow.set_experiment("XGBoost Potability Experiment")

    study = optuna.create_study(direction='minimize')
    study.optimize(optimize_model, timeout=300)

    best_trial = study.best_trial

    mlflow.log_params(best_trial.params)
    mlflow.log_metric("best_valid_mae", best_trial.value)

    # Guardar el mejor modelo
    best_params = best_trial.params
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact('models/best_model.pkl')

    # Guardar gráficos de Optuna
    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.write_image("plots/optimization_history.png")
    
    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.write_image("plots/param_importances.png")
    
    mlflow.log_artifact('plots/optimization_history.png')
    mlflow.log_artifact('plots/param_importances.png')
    
if __name__ == "__main__":
    main()