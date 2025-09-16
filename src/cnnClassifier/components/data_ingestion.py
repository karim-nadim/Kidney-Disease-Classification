import os
import zipfile
import urllib.request as request
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories



class DataIngestion:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
        create_directories([self.config.data_ingestion.root_dir])

        

    
    def download_file(self):
        if not os.path.exists(self.config.data_ingestion.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.data_ingestion.source_URL,
                filename = self.config.data_ingestion.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.data_ingestion.local_data_file))}")
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.data_ingestion.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.data_ingestion.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

