import cv2
import pytesseract
import os
import mlflow
import mlflow.sklearn
from flask import Flask, render_template

from config import INPUT_DIR, OUTPUT_DIR, TRACKING_URI, EXPERIMENT_NAME

app = Flask(__name__, template_folder="../template")

def text_recognition():
    input_file_path = os.path.join(INPUT_DIR, "hello-world.jpeg")
    output_file_path = os.path.join(OUTPUT_DIR, "recognized.txt")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        try:
            if not os.path.exists(input_file_path):
                print(f"Error: The file {input_file_path} was not found!")
                mlflow.log_param("file_found", False)
                mlflow.log_param("status", "failed")
                return

            mlflow.log_param("file_found", True)

            # Read image and process
            gray = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
            extracted_text = pytesseract.image_to_string(gray, lang='eng+jpn')

            # Log extracted text length
            mlflow.log_metric("text_length", len(extracted_text))

            # Save to a text file
            with open(output_file_path, "w", encoding="utf-8") as file:
                file.write(extracted_text)
            print(f"Extracted text saved to {output_file_path}")

            mlflow.log_artifacts(output_file_path, artifact_path="artifacts")
            print(f"Artifact saved under: {mlflow.get_artifact_uri()}")

        except FileNotFoundError as e:
            mlflow.log_param("error", "FileNotFoundError")
            mlflow.log_param("status", "failed")
            print(f"File not found: {e}")
        except ValueError as e:
            mlflow.log_param("error", "ValueError")
            mlflow.log_param("status", "failed")
            print(f"Value error: {e}")
        except Exception as e:
            mlflow.log_param("error", "GeneralException")
            mlflow.log_param("status", "failed")
            print(f"An unexpected error occurred: {e}")

def text_reading():
    input_file_path = os.path.join(OUTPUT_DIR, "recognized.txt")

    with mlflow.start_run():
        try:
            if not os.path.exists(input_file_path):
                print(f"Error: The file {input_file_path} was not found!")
                mlflow.log_param("text_file_found", False)
                mlflow.log_param("status", "failed")
                return

            mlflow.log_param("text_file_found", True)

            # Read and log extracted text
            with open(input_file_path, 'r', encoding="utf-8") as file:
                text = file.read()
                mlflow.log_metric("text_file_length", len(text))
                print(f"Reading text from file {input_file_path}")
                print(text)
                mlflow.log_param("status", "success")

        except FileNotFoundError as e:
            mlflow.log_param("error", "FileNotFoundError")
            mlflow.log_param("status", "failed")
            print(f"File not found: {e}")
        except ValueError as e:
            mlflow.log_param("error", "ValueError")
            mlflow.log_param("status", "failed")
            print(f"Value error: {e}")
        except Exception as e:
            mlflow.log_param("error", "GeneralException")
            mlflow.log_param("status", "failed")
            print(f"An unexpected error occurred: {e}")
@app.route('/')
def display_results():
    input_image_path = os.path.join(INPUT_DIR, "hello-world.jpeg")
    output_text_path = os.path.join(OUTPUT_DIR, "recognized.txt")

    extracted_text = ""
    if os.path.exists(output_text_path):
        with open(output_text_path, "r", encoding="utf-8") as file:
            extracted_text = file.read()

    from_net="https://m.media-amazon.com/images/M/MV5BYWJkNGIzMmQtZDY4Ni00OTdmLWExNjUtMTM1OWU4MGRhNDM5XkEyXkFqcGc@._V1_.jpg"
    return render_template('index.html',
                           input_image=from_net, extracted_text=extracted_text)

if __name__ == "__main__":
    text_recognition()
    text_reading()
    app.run(debug=True, host='0.0.0.0', port=8080)
