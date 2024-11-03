import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlflow.tracking import MlflowClient

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print Python and MLflow versions
import sys
logger.info(f"Python version: {sys.version}")
logger.info(f"MLflow version: {mlflow.__version__}")

mlflow.set_tracking_uri("http://localhost:5002")
logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

experiment_name = "random-forest-hyperopt"

# Create or get the experiment
client = MlflowClient()
try:
    experiment = client.create_experiment(experiment_name)
    logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment}")
except mlflow.exceptions.MlflowException as e:
    if "already exists" in str(e):
        experiment = client.get_experiment_by_name(experiment_name)
        logger.info(f"Experiment '{experiment_name}' already exists with ID: {experiment.experiment_id}")
    else:
        logger.error(f"Error creating/getting experiment: {e}")
        raise

mlflow.set_experiment(experiment_name)

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def run_optimization(data_path: str = "./output", num_trials: int = 15):
    logger.info(f"Data path: {data_path}")
    logger.info(f"Number of trials: {num_trials}")

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    best_rmse = float('inf')

    def objective(params):
        nonlocal best_rmse
        with mlflow.start_run(nested=True):
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            
            mlflow.log_params(params)
            mlflow.log_metric("rmse", rmse)
            
            logger.info(f"RMSE: {rmse}")

            if rmse < best_rmse:
                best_rmse = rmse

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    
    try:
        with mlflow.start_run():
            fmin(
                fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=num_trials,
                trials=Trials(),
                rstate=rstate
            )
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}")
        raise

    print(f"Best RMSE: {best_rmse}")
    return best_rmse

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization_cli(data_path: str, num_trials: int):
    return run_optimization(data_path, num_trials)

if __name__ == '__main__':
    run_optimization()