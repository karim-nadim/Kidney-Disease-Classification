from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        prepare_base_model = PrepareBaseModel()
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == '__main__':
    STAGE_NAME = "Prepare base model"
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e