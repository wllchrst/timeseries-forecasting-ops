FROM python:3.11-alpine

# Install required system dependencies
RUN apk add --no-cache git gcc g++ musl-dev libffi-dev

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD ["python", "-m", "main"]
