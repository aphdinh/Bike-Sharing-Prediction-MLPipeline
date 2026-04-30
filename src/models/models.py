import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.early_stop import no_progress_loss
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

def status(msg):
    logging.info(msg)

def get_models():
    return {
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=15),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, random_state=42, max_depth=15, learning_rate=0.1,
            num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, verbose=-1
        ),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1, verbosity=0)
    }

def create_model(model_name, params=None):
    models = get_models()
    if params:
        return type(models[model_name])(**params)
    return models[model_name]

def get_lightgbm_param_grid():
    return {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [6, 10, 15, 20, -1],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [15, 31, 63, 127],
        'feature_fraction': [0.6, 0.8, 1.0],
        'bagging_fraction': [0.6, 0.8, 1.0],
        'bagging_freq': [0, 5, 10],
        'min_child_samples': [10, 20, 30],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

def get_xgboost_param_grid():
    return {
        'n_estimators': [100, 200],  # Reduced from [100, 200, 300]
        'max_depth': [3, 6],         # Reduced from [3, 6, 9, 12]
        'learning_rate': [0.1, 0.2], # Reduced from [0.01, 0.1, 0.2]
        'subsample': [0.8, 1.0],     # Reduced from [0.8, 0.9, 1.0]
        'colsample_bytree': [0.8, 1.0], # Reduced from [0.8, 0.9, 1.0]
        'reg_alpha': [0, 0.1],       # Reduced from [0, 0.1, 0.5]
        'reg_lambda': [0.1, 1]       # Reduced from [0.1, 1, 10]
    }

def get_extra_trees_param_grid():
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

def get_param_grid(model_name):
    grids = {
        'LightGBM': get_lightgbm_param_grid(),
        'XGBoost': get_xgboost_param_grid(),
        'Extra Trees': get_extra_trees_param_grid()
    }
    return grids.get(model_name)

def create_base_model(model_name):
    models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0),
        'Extra Trees': ExtraTreesRegressor(random_state=42, n_jobs=-1)
    }
    return models.get(model_name)

def log_tuning_params(mlflow, param_grid, method="RandomizedSearchCV"):
    mlflow.log_param("tuning_method", method)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("n_iter", 50)
    mlflow.log_param("param_grid", str(param_grid))

def save_cv_results(random_search, mlflow):
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_df.to_csv("cv_results.csv", index=False)
    mlflow.log_artifact("cv_results.csv", artifact_path="hyperparameter_tuning")
    os.remove("cv_results.csv")

def hyperparameter_tuning_with_mlflow(X_train, y_train, model_name):
    from sklearn.model_selection import RandomizedSearchCV
    import time
    
    param_grid = get_param_grid(model_name)
    if not param_grid:
        status(f"No parameter grid found for {model_name}")
        return None
    
    base_model = create_base_model(model_name)
    if not base_model:
        status(f"No base model found for {model_name}")
        return None
    
    # Add progress monitoring
    print(f"🔍 Starting hyperparameter tuning for {model_name}")
    print(f"   Parameter combinations: {len(param_grid)}")
    print(f"   CV folds: 3")
    print(f"   Total fits: {len(param_grid) * 3}")
    print(f"   Estimated time: ~{len(param_grid) * 3 * 2} seconds")
    
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=10,  # Reduced from 50
        cv=3,       # Reduced from 5
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1   # Add progress output
    )
    
    with mlflow.start_run(run_name=f"{model_name}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        log_tuning_params(mlflow, param_grid)
        
        print(f"⏳ Fitting {model_name} with RandomizedSearchCV...")
        random_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Tuning completed in {elapsed_time:.1f} seconds")
        
        # Log results
        mlflow.log_metric("best_cv_score", -random_search.best_score_)
        mlflow.log_params(random_search.best_params_)
        
        save_cv_results(random_search, mlflow)

        status(f"Best {model_name} params: {random_search.best_params_}")
        status(f"Best CV score: {-random_search.best_score_:.4f}")

        return random_search.best_estimator_

def get_lightgbm_search_space():
    return {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500, 800]),
        'max_depth': hp.choice('max_depth', [6, 10, 15, 20, -1]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'num_leaves': hp.choice('num_leaves', [15, 31, 63, 127, 255]),
        'feature_fraction': hp.uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1.0),
        'bagging_freq': hp.choice('bagging_freq', [0, 1, 5, 10]),
        'min_child_samples': hp.choice('min_child_samples', [5, 10, 20, 30, 50]),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0)),
        'min_split_gain': hp.uniform('min_split_gain', 0.0, 1.0),
        'subsample_for_bin': hp.choice('subsample_for_bin', [50000, 100000, 200000])
    }

def get_xgboost_search_space():     
    return {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500, 800]),
        'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8, 9]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.6, 1.0),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0)),
        'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7]),
        'gamma': hp.uniform('gamma', 0.0, 0.5)
    }

def get_random_forest_search_space():
    return {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500]),
        'max_depth': hp.choice('max_depth', [10, 15, 20, 25, None]),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10, 15]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4, 8]),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.3, 0.6, 0.8]),
        'bootstrap': hp.choice('bootstrap', [True, False]),
        'max_samples': hp.uniform('max_samples', 0.7, 1.0)
    }

def get_search_space(model_name):
    spaces = {
        'LightGBM': get_lightgbm_search_space(),
        'XGBoost': get_xgboost_search_space(),
        'Random Forest': get_random_forest_search_space()
    }
    return spaces.get(model_name)

def create_model_from_params(model_name, params):
    models = {
        'LightGBM': lambda p: lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1, **p),
        'XGBoost': lambda p: xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1, **p),
        'Random Forest': lambda p: RandomForestRegressor(random_state=42, n_jobs=-1, **p)
    }
    
    model_creator = models.get(model_name)
    return model_creator(params) if model_creator else None

def create_objective_function(X_train, y_train, X_val, y_val, model_name):
    def objective(params):
        try:
            model = create_model_from_params(model_name, params)
            if not model:
                return {'loss': float('inf'), 'status': STATUS_OK}
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Safe metrics calculation
            try:
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                # Validate metrics
                if not np.isfinite(rmse) or np.isnan(rmse):
                    rmse = float('inf')
                if not np.isfinite(r2) or np.isnan(r2):
                    r2 = 0.0
            except:
                rmse = float('inf')
                r2 = 0.0
            
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("validation_rmse", rmse)
                mlflow.log_metric("validation_r2", r2)
                
            return {'loss': rmse, 'status': STATUS_OK}
        except Exception as e:
            status(f"Error in objective function: {e}")
            return {'loss': float('inf'), 'status': STATUS_OK}
    
    return objective

def save_trials_data(trials, mlflow):
    trials_df = pd.DataFrame([
        {**trial['misc']['vals'], 'loss': trial['result']['loss']} 
        for trial in trials.trials
    ])
    trials_df.to_csv("hyperopt_trials.csv", index=False)
    mlflow.log_artifact("hyperopt_trials.csv", artifact_path="hyperparameter_tuning")
    os.remove("hyperopt_trials.csv")

def create_hyperopt_convergence_plot(trials, model_name):
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    
    losses = [trial['result']['loss'] for trial in trials.trials]
    best_losses = [min(losses[:i+1]) for i in range(len(losses))]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'o-', alpha=0.6, markersize=4)
    plt.xlabel('Trial')
    plt.ylabel('Validation RMSE')
    plt.title(f'{model_name}: All Trials')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(best_losses, 'g-', linewidth=2)
    plt.xlabel('Trial')
    plt.ylabel('Best Validation RMSE')
    plt.title(f'{model_name}: Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("hyperopt_convergence.png", dpi=300, bbox_inches='tight')
    mlflow.log_artifact("hyperopt_convergence.png", artifact_path="plots")
    plt.close()
    os.remove("hyperopt_convergence.png")

def hyperparameter_tuning_with_hyperopt(X_train, y_train, X_val, y_val, model_name, max_evals=100):
    with mlflow.start_run(run_name=f"Hyperopt_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        status(f"Hyperopt tuning for {model_name}...")
        
        search_space = get_search_space(model_name)
        if not search_space:
            status(f"Hyperopt tuning not implemented for {model_name}")
            return None, None, None
        
        mlflow.log_param("tuning_method", "Hyperopt_TPE")
        mlflow.log_param("max_evals", max_evals)
        mlflow.log_param("validation_samples", len(X_val))
        mlflow.log_param("search_space", str(search_space))
        
        objective = create_objective_function(X_train, y_train, X_val, y_val, model_name)
        
        trials = Trials()
        status("Starting Hyperopt optimization...")
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            early_stop_fn=no_progress_loss(30),
            verbose=True,
            rstate=np.random.default_rng(42)
        )
        
        best_params = space_eval(search_space, best)
        best_loss = min([trial['result']['loss'] for trial in trials.trials])
        
        for param, value in best_params.items():
            mlflow.log_param(f"best_{param}", value)
        
        mlflow.log_metric("best_validation_rmse", best_loss)
        mlflow.log_metric("n_trials", len(trials.trials))
        
        save_trials_data(trials, mlflow)
        create_hyperopt_convergence_plot(trials, model_name)
        
        status(f"Best parameters: {best_params}")
        status(f"Best validation RMSE: {best_loss:.4f}")
        
        best_model = create_model_from_params(model_name, best_params)
        if best_model:
            best_model.fit(X_train, y_train)
        
        return best_model, best_params, trials

def evaluate_tuning_method(X_train, y_train, X_val, y_val, model_name, tuning_func, method_name):
    status(f"\n{method_name}...")
    start_time = datetime.now()
    
    if method_name == "RandomizedSearchCV":
        best_model = tuning_func(X_train, y_train, model_name)
    else:
        best_model, _, _ = tuning_func(X_train, y_train, X_val, y_val, model_name)
    
    tuning_time = (datetime.now() - start_time).total_seconds()
    
    if best_model:
        y_pred = best_model.predict(X_val)
        try:
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            if not np.isfinite(rmse) or np.isnan(rmse):
                rmse = float('inf')
            if not np.isfinite(r2) or np.isnan(r2):
                r2 = 0.0
        except:
            rmse = float('inf')
            r2 = 0.0
    else:
        rmse = float('inf')
        r2 = 0.0
    
    return best_model, rmse, r2, tuning_time

def log_comparison_results(mlflow, random_results, hyperopt_results, model_name):
    random_model, random_rmse, random_r2, random_time = random_results
    hyperopt_model, hyperopt_rmse, hyperopt_r2, hyperopt_time = hyperopt_results
    
    with mlflow.start_run(run_name=f"HPO_Comparison_{model_name}"):
        mlflow.log_metric("random_search_rmse", random_rmse)
        mlflow.log_metric("random_search_r2", random_r2)
        mlflow.log_metric("random_search_time", random_time)
        mlflow.log_metric("hyperopt_rmse", hyperopt_rmse)
        mlflow.log_metric("hyperopt_r2", hyperopt_r2)
        mlflow.log_metric("hyperopt_time", hyperopt_time)
        mlflow.log_metric("rmse_improvement", random_rmse - hyperopt_rmse)
        mlflow.log_metric("r2_improvement", hyperopt_r2 - random_r2)
        mlflow.log_metric("time_ratio", hyperopt_time / random_time)

def display_comparison_results(random_results, hyperopt_results):
    random_model, random_rmse, random_r2, random_time = random_results
    hyperopt_model, hyperopt_rmse, hyperopt_r2, hyperopt_time = hyperopt_results
    
    status(f"\n=== Results Comparison ===")
    print(f"\n=== Results Comparison ===")
    print(f"RandomizedSearchCV - RMSE: {random_rmse:.4f}, R²: {random_r2:.4f}, Time: {random_time:.1f}s")
    print(f"Hyperopt           - RMSE: {hyperopt_rmse:.4f}, R²: {hyperopt_r2:.4f}, Time: {hyperopt_time:.1f}s")
    print(f"Improvement        - RMSE: {random_rmse - hyperopt_rmse:.4f}, R²: {hyperopt_r2 - random_r2:.4f}")

def hyperparameter_comparison(X_train, y_train, X_val, y_val, model_name='LightGBM'):
    status(f"\n=== Hyperparameter Tuning Comparison for {model_name} ===")
    
    random_results = evaluate_tuning_method(
        X_train, y_train, X_val, y_val, model_name, 
        hyperparameter_tuning_with_mlflow, "RandomizedSearchCV"
    )
    
    hyperopt_results = evaluate_tuning_method(
        X_train, y_train, X_val, y_val, model_name, 
        hyperparameter_tuning_with_hyperopt, "Hyperopt"
    )
    
    log_comparison_results(mlflow, random_results, hyperopt_results, model_name)
    display_comparison_results(random_results, hyperopt_results)
    
    random_model, random_rmse, _, _ = random_results
    hyperopt_model, hyperopt_rmse, _, _ = hyperopt_results
    
    return hyperopt_model if hyperopt_rmse < random_rmse else random_model 