from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        evaluation = Evaluation()
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == '__main__':
    STAGE_NAME = "Evaluation stage"
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e