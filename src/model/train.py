import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import mlflow
from mlflow import MlflowClient
from src import init_mlflow


@click.command()
@click.argument("train_path", type=click.Path(exists=True))
@click.argument("model_output_path", type=click.Path())
@click.argument("n_estimators", type=int, default=100)
def train(train_path: str, model_output_path: str, n_estimators: int):
    """
    Обучение RandomForestClassifier и сохранение модели в MLflow Model Registry (Staging).

    TRAIN_PATH: путь к train CSV
    MODEL_OUTPUT_PATH: путь для сохранения модели
    N_ESTIMATORS: количество деревьев в лесу
    """
    # Инициализация mlflow
    init_mlflow()
    mlflow.set_experiment("auto_ml_experiment")

    # Загружаем данные
    df = pd.read_csv(train_path)
    X = df.drop(columns=['target'])
    y = df['target']

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    with mlflow.start_run():
        model.fit(X, y)

        # Логируем гиперпараметры и метрики
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("train_samples", len(df))

        # Локально сохраняем модель
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="IrisClassifier"
        )

        client = MlflowClient()

        # Проверяем, зарегистрирована ли модель
        try:
            client.get_registered_model("IrisClassifier")
        except mlflow.exceptions.RestException:
            client.create_registered_model("IrisClassifier")

        # Получаем последнюю версию модели
        versions = client.search_model_versions("name='IrisClassifier'")
        latest_version = max(int(v.version) for v in versions)

        # Переводим последнюю версию в Staging
        client.transition_model_version_stage(
            name="IrisClassifier",
            version=latest_version,
            stage="Staging"
        )
        click.echo("Проерка модели в MLflow Model Registry пройдена.")
        click.echo(f"✅ Модель IrisClassifier v{latest_version} сохранена и переведена в Staging.")

    click.echo(f"Модель сохранена локально в {model_output_path} и залогирована в MLflow.")


if __name__ == "__main__":
    train()
