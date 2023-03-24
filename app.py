from configparser import ConfigParser
from src.train import Train
from src.utils.utils import Functional
config = ConfigParser()
config.read("./config/config.ini")

if __name__ == '__main__':
    Functional.save_config(config , )
    Train.train(config)