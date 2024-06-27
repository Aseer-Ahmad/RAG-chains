import os
import sys

def setup_env():
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['GOOGLE_API_KEY'] = None # add your API key here
    os.environ['LANGCHAIN_API_KEY'] = None # add your API key here