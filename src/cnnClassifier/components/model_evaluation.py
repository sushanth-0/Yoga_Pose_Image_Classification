import tensorflow as tf
from pathlib import Path
import mlflow
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
import logging


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        logging.info("Validation data generator created successfully.")

    @staticmethod
    def load_model(path: Path) -> tf.keras.models.Model:
        try:
            model = tf.keras.models.load_model(path)
            logging.info(f"Model loaded successfully from {path}.")
            return model
        except Exception as e:
            logging.error(f"Error loading model from {path}: {e}")
            raise

    def evaluation(self):
        try:
            self.model = self.load_model(self.config.path_of_model)
            self._valid_generator()
            self.score = self.model.evaluate(self.valid_generator)
            self.save_score()
            logging.info(
                f"Model evaluation completed with score: {self.score}.")
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise

    def save_score(self):
        try:
            scores = {"loss": self.score[0], "accuracy": self.score[1]}
            save_json(path=Path("scores.json"), data=scores)
            logging.info("Scores saved successfully to scores.json.")
        except Exception as e:
            logging.error(f"Error saving scores: {e}")
            raise

    def log_into_mlflow(self):
        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics(
                    {"loss": self.score[0], "accuracy": self.score[1]})

                # Save the model locally using TensorFlow's model.save
                local_model_path = "model"
                self.model.save(local_model_path)

                # Log the model artifacts manually in MLflow
                mlflow.log_artifacts(local_model_path, artifact_path="model")

                if tracking_url_type_store != "file":
                    mlflow.register_model(
                        "runs:/{}/model".format(mlflow.active_run().info.run_id), "VGG16Model")

            logging.info("Model logged to MLflow successfully.")
        except Exception as e:
            logging.error(f"Error logging model to MLflow: {e}")
            raise
