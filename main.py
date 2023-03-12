""" This Project is based on Artin Zareie's work <https://ipm.ssaa.ir/Search-Result?page=1&DecNo=139950140003004932&RN=105750>  """

# Prevent tensorflow logs
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--verbose", choices=[0, 1], type=int, nargs=1, default=0, help="Show all tensorflow logs")

args = parser.parse_args()

if args.verbose == 0:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import ArNetModel
from terminal_utils import PrintColored

if __name__ == "__main__":

    cprint = PrintColored()

    cprint("\n\n\tWelcome to this application, This application is a client for ArNet architecture by Artin Zareie (https://ipm.ssaa.ir/Search-Result?page=1&DecNo=139950140003004932&RN=105750).\n This architecture uses Convolutional layers and the flatten input itself to train on pictures and classify them. Also it is possible to use this model for regression and other applications, but this speciefic python program is designed for classification problems.\n\n", color=cprint.CYAN_DARK)
    
    # model = ArNetModel((32, 32, 3), 10)
    # model.plot_model()
    # model.summary()
