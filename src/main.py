import cv2
import pytesseract
import os

from config import INPUT_DIR, OUTPUT_DIR

def text_recognition():
    input_file_path = os.path.join(INPUT_DIR, "hello-world.jpeg")
    output_file_path = os.path.join(OUTPUT_DIR, "recognized.txt")

    try:
        if not os.path.exists(input_file_path):
           print(f"Error: The file {input_file_path} was not found!")

        gray = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
        extracted_text = pytesseract.image_to_string(gray, lang='eng+jpn')  # English and Japanese support

        # Save to a text file
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(extracted_text)

        print(f"Extracted text saved to {output_file_path}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def text_reading():
    input_file_path = os.path.join(OUTPUT_DIR, "recognized.txt")

    try:
        if not os.path.exists(input_file_path):
           print(f"Error: The file {input_file_path} was not found!")
        else:
            print(f"Reading text from file {input_file_path}")
            print(open(input_file_path, 'r').read())

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    text_recognition()
    text_reading()
