FROM python:3.10

WORKDIR /app

# Copy requirements file first to improve caching
COPY requirements.txt .
COPY supervisord.conf /etc/supervisord.conf

# Ensure the `app` folder exists before copying
RUN mkdir -p /app

# Copy the entire app directory
COPY ./app /app
COPY ./mlflow /app/mlflow
COPY ./notebooks /app/notebooks

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000 8888 5000

# Run FastAPI application
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
