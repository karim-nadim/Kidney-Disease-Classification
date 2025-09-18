import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import dagshub

class Evaluation:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.mlflow_uri = "https://dagshub.com/karim-nadim/Kidney-Disease-Classification.mlflow"
        
        create_directories([self.config.artifacts_root])

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.params.IMAGE_SIZE[:-1],
            batch_size=self.params.BATCH_SIZE,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=Path(os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")),
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.training.trained_model_path)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

        
    def log_into_mlflow(self):
        # Make sure DagsHub is initialized first
        dagshub.init(
            repo_owner='karim-nadim',
            repo_name='Kidney-Disease-Classification',
            mlflow=True
        )

        # Optional: set explicit registry URI
        mlflow.set_registry_uri(self.mlflow_uri)

        # Debug: confirm URI
        print("Tracking URI:", mlflow.get_tracking_uri())
        print("Registry URI:", mlflow.get_registry_uri())

        with mlflow.start_run():
            mlflow.log_params(self.params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })

            # Always try to register the model on DagsHub
            try:
                mlflow.keras.log_model(
                    self.model,
                    artifact_path="model",
                    registered_model_name="VGG16Model"
                )
            except Exception as e:
                print("⚠️ Could not register model, falling back to artifact only:", e)
                mlflow.keras.log_model(self.model, artifact_path="model")

