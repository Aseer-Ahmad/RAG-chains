import os

def setup_env():
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = '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'
    os.environ['OPENAI_API_KEY'] = 'sk-proj-LzYCqVZM8srcVuhTYYO2T3BlbkFJ9LVFcGprgUS7NPo5Dwue'
