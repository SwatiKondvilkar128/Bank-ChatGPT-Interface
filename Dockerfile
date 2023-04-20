# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster
# Set the working directory to /app
WORKDIR /app
# Copy the requirements file into the container at /app
COPY pip install -r requirements.txt .
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the current directory contents into the container at /app
COPY . app
# Expose the port that the Flask app will run on
EXPOSE 5000
# Run the command to start the Flask app
CMD [ "python", "Flask_app.py" ]
