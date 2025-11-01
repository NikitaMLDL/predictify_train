from dotenv import load_dotenv
import os
import mlflow


def init_mlflow():
    """
    Инициализация MLflow для всего пайплайна.
    Подхватывает переменные из .env или из окружения системы.
    """
    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
