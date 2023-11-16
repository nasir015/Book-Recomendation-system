import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from pipeline.logging import logger
from pipeline.Exception import CustomException
import pickle


path = open("Log\\common.txt", "w")
log_path= 'Log\\common.txt'


logger(log_path,'save_object function is started')
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    logger(log_path,'save_object function is completed')




def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        

    except Exception as e:
        raise CustomException(e, sys)
logger(log_path,'load_object function is completed')

