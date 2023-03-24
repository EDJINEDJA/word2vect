from configparser import ConfigParser
from src.train import Train
from src.utils.utils import Functional
import os
config = ConfigParser()
config.read("./config/config.ini")

if __name__ == '__main__':
    try:
        os.makedirs(config.get("hyperparameters","model_dir"))
    except FileExistsError :
        pass
    Functional.save_config(config )
    Train.train(config)