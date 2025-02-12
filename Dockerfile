# Use official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for MLflow UI (if needed)
EXPOSE 5001

# Set the command to run your script
CMD ["python", "main.py"]
