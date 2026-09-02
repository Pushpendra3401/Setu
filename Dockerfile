# Use an official Python image
FROM python:3.11-slim

# Set the folder inside the server where code will live
WORKDIR /app

# Copy the requirements list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code files
COPY . .

# Tell Render which port we use
EXPOSE 8000

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
