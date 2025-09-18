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


if __name__ == '__main__':
    STAGE_NAME = "Training"
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e