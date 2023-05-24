import os
import pickle
import click
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
#
# mlflow.set_artifact_uri("~/Documents/MlOPS-bootcamp/experiment tracking/ARTIFACT")

# Create the experiment if it does not exist
experiment_name = "random forest -optuna-new"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)



def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=10,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(trial):
        with mlflow.start_run():
          

          params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
            'random_state': 42,
            'n_jobs': -1
          }

          rf = RandomForestRegressor(**params)
          rf.fit(X_train, y_train)
          y_pred = rf.predict(X_val)
          rmse = mean_squared_error(y_val, y_pred, squared=False)

    #Log the RMSE to the tracking server
          trial.set_user_attr('rmse', rmse)
          
          mlflow.log_artifact('/home/mussie/Documents/MlOPS-bootcamp/experiment tracking/ARTIFACT')
       

          return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':
    run_optimization()