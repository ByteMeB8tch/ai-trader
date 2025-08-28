# Use a lightweight Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the bot when the container starts
CMD ["python", "paper_trade.py"]
