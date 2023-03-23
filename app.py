from configparser import ConfigParser
from src.train import Train
config = ConfigParser()
config.read("./config/config.ini")

if __name__ == '__main__':
    Train.train(config)