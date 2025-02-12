# Use official Python image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy entire project into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the correct working directory for main.py
WORKDIR /app/src

# Expose the port for MLflow UI (if needed)
EXPOSE 5001

# Run the application
CMD ["python", "main.py"]

