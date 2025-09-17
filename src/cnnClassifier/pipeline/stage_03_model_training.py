from cnnClassifier.components.model_training import Training
from cnnClassifier import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        training = Training()
        training.get_base_model()
        training.train_valid_generator()
        training.train()