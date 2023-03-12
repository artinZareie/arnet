""" This Project is based on Artin Zareie's work <https://ipm.ssaa.ir/Search-Result?page=1&DecNo=139950140003004932&RN=105750>  """

# Prevent tensorflow logs
import os
import argparse
import inquirer
import re

parser = argparse.ArgumentParser()

parser.add_argument("-v", "--verbose", choices=[0, 1], type=int, nargs=1, default=0, help="Show all tensorflow logs")

args = parser.parse_args()

if args.verbose == 0:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import ArNetModel
from terminal_utils import PrintColored, clear_terminal

if __name__ == "__main__":

    cprint = PrintColored()

    cprint("\n\n\tWelcome to this application, This application is a client for ArNet architecture by Artin Zareie (https://ipm.ssaa.ir/Search-Result?page=1&DecNo=139950140003004932&RN=105750).\n This architecture uses Convolutional layers and the flatten input itself to train on pictures and classify them. Also it is possible to use this model for regression and other applications, but this speciefic python program is designed for classification problems.\n\n", color=cprint.CYAN_DARK)

    cprint("Program is going to ask you some information needed by model\n\n", color=cprint.YELLOW)

    available_activations = ['relu', 'elu', 'sigmoid', 'tanh', 'linear']
    available_losses = ['categorical_crossentropy', 'mse']
    available_optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta']

    model_questions = [
        inquirer.Text(name='width', message="Enter picture's width", validate=lambda _,x: re.match(r"^[0-9 -]+$", x), default="32"),
        inquirer.Text(name='height', message="Enter picture's height", validate=lambda _, x: re.match(r"^[0-9 -]+$", x), default="32"),
        inquirer.Text(name='depth', message="How many channels does picture have?", validate=lambda _, x: re.match(r"^[0-9 -]+$", x), default="3"),
        inquirer.Text(name='output', message="How many categories do you have?", validate=lambda _, x: re.match(r"^[0-9 -]+$", x), default="10"),
        inquirer.List(name='activation', message="Which activation function will you use?", choices=available_activations),
        inquirer.Text(name='dropout', message="Dropout should be set at what rate?", default='0.2', validate=lambda _, x: re.match(r"^0\.[0-9]+]?$", x)),
        inquirer.List(name='padding', message="Which type of padding will you use?", choices=['same', 'valid']),
    ]

    model_preferences = inquirer.prompt(model_questions)

    model = ArNetModel((int(model_preferences['width']), int(model_preferences['height']), int(model_preferences['depth'])), int(model_preferences['output']), dropout_rate=float(model_preferences['dropout']), activation=model_preferences['activation'], padding=model_preferences['padding'])

    selection_message = "Choose what do you want to do right now?"
    options = ["Train model on images", "Load a trained model", "Show model's architecture", "Show model's summary", "Exit"]
    
    questions = [
        inquirer.List('choice', message=selection_message, choices=options)
    ]

    while True: 
        answers = inquirer.prompt(questions)
        selection = options.index(answers['choice'])

        if selection == 0:
            
            print("\n\n")
            cprint("\tWelcome to training section. In this section, you'll have to enter a dataset and it's information to the app, and program will generate a trained model in models directory.\n" + "\tTo begin, you're gonna need to place pictures in training folder. I must be structed like this:\n\t+ Each category or class must have a folder. folder's names indicate categories name.\n\t+ in Each category folder, put images related to that category.", color=cprint.BLUE_DARK)
            cprint("\nIf you haven't put your dataset in your training folder yet, terminate the program using CTRL+C and put dataset first.\n\n", color=cprint.YELLOW)

            train_questions = [
                inquirer.List(name='loss', message="Which loss function will you use?", choices=available_losses),
                inquirer.List(name='optimizer', message="How do you like your model to be optimized?", choices=available_optimizers),
                inquirer.List(name='batch_size', message="How much data should be trained on a single step of training?", validate=lambda _, x: re.match(r"^[0-9 -]+$", x), default="32"),
                inquirer.List(name='epochs', message="How many epochs do you need to train your data?", validate=lambda _, x: re.match(r"^[0-9 -]+$", x), default="10"),
            ]

        elif selection == 1:
            pass
        elif selection == 2:
            model.plot_model()
            cprint(f"\n\nImage is saved in {os.getcwd()}{os.sep}model_architecture.png\n\n", color=cprint.GREEN)

        elif selection == 3:
            print("\n")
            model.summary()
            print("\n")

        elif selection == 4:
            break

    print("Goodbye!")
    