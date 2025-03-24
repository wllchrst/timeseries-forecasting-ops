"""Main Script for running the project
"""
import uvicorn
from api import app
from helper import EnvHelper

if __name__ == '__main__':
    env_helper = EnvHelper()
    uvicorn.run(app, host=env_helper.api_host)
