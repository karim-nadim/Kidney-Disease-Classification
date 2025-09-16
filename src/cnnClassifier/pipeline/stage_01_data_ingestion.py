from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        data_ingestion = DataIngestion()
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
