FROM python:3.11-slim 

RUN apt update && apt install -y git gcc g++ musl-dev libffi-dev

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD ["python", "-m", "main"]
