from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        prepare_base_model = PrepareBaseModel()
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

