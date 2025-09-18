from cnnClassifier.components.model_evaluation_mlflow import Evaluation
from cnnClassifier import logger

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        evaluation = Evaluation()
        evaluation.evaluation()
        evaluation.log_into_mlflow()

            