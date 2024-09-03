import openai
# from langchain_openai import ChatOpenAI
from typing import Any
from openai import OpenAI
from flask import session
from langchain_openai import OpenAIEmbeddings

class OpenAiService:
    
    def check_if_api_key_is_valid(api_key: str) -> bool :
        """Methd to check if provided key is valid

        Args:
            api_key (str):

        Returns:
            bool:
        """
        try:
            openai.api_key=api_key
            #Try to list available models as a test
            models = openai.models.list()
            print(models)
            if models:
                return True
            return False
        except openai.APIError:
            return False

    def get_embedding(self, data: str) -> Any:
        """
        Returns vector for provided data
        :param data: string
        :return: vector
1        """
        client = self.get_client()
        
        print(f" to idzie do embeddowania{data}")
        print(type(data))
        return client.embeddings.create(input=[data], model="text-embedding-3-small").data[0].embedding
    
    def get_client(self):
        return OpenAI(api_key=session[session['username'] + 'api_key'])
    
    def get_embeddings_model(self):
        return OpenAIEmbeddings(model="text-embedding-3-small",
                                  openai_api_key=session[session['username'] + 'api_key'])