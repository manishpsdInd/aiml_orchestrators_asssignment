import os

from mlflow.experiments import EXPERIMENT_ID

# Define the root path of the project dynamically
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# Define paths for input and output directories relative to the root path
INPUT_DIR = os.path.join(ROOT_PATH, "data/input")
OUTPUT_DIR = os.path.join(ROOT_PATH, "data/output")

TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "Text Recognition Experiment"
EXPERIMENT_ID = "ECL542.I"
