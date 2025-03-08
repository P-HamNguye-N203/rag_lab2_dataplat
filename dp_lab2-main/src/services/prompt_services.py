# services/prompt_services.py
from qdrant_client import QdrantClient
from src.config.settings import HTTP_500_INTERNAL_SERVER_ERROR
from src.config.settings import MAX_NO_SEARCH_RESULTS_QDRANT
from src.config.settings import QDRANT_COLLECTION
from qdrant_client import QdrantClient
from google import genai
import google.generativeai as gg_genai


class PromptServices:
    """
    Class handling various prompt services related operations.
    """

    # Constructor
    def __init__(self,
                 openai_api_key,
                 qdrant_url,
                 qdrant_api_key,
                 openai_embedding_model,
                 openai_gpt_model):

        self._open_api_key = openai_api_key
        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key
        self._openai_embedding_model = openai_embedding_model
        self._openai_gpt_model = openai_gpt_model
        gg_genai.configure(api_key=self._open_api_key)
        self._openai_client = genai.Client(api_key= self._open_api_key)
        self._model = gg_genai.GenerativeModel(self._openai_gpt_model)
        self._qdrant_client = QdrantClient(url=self._qdrant_url,
                                           api_key=self._qdrant_api_key)
        self._system_prompt = ("You are a knowledgeable assistant. "
                               "Please use the provided context to answer the question. "
                               "Please be as helpful and relevant as possible. "
                               "If you do not have the information, "
                               "please do not make up the answer.")

    #############################################
    # Get the embedding of the query using the provided embedding model
    def get_embedding(self, query):

        # Get the embedding of the query
        try:
            embedding_response = self._openai_client.models.embed_content(
                model=self._openai_embedding_model,
                contents= query
            )
        except Exception as e:
            print("Exception :", str(e))

        return embedding_response.embeddings[0].values

    #############################################
    # Search for closest texts in Qdrant vector database
    def get_context(self, embedding):

        try:
            search_results = self._qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=embedding,
                limit=MAX_NO_SEARCH_RESULTS_QDRANT
            )
        except Exception as e:
            print("Exception :", str(e))

        return search_results

    #############################################
    # Search for closest texts in Qdrant vector database
    def get_context_2(self, embedding):

        try:
            search_results = self._qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=embedding,
                limit=1
            )
        except Exception as e:
            print("Exception :", str(e))

        return search_results   
    #############################################
    def get_response(self, query, search_results):

        context = ""
        for result in search_results:
            context += result.payload['text'] + "\n"

        try:
            chat = self._model.start_chat()
            chat_response = chat.send_message(
                {
                    "parts": [
                        {"text": self._system_prompt},
                        {"text": context},
                        {"text": query}
                    ]
                }
            )
        except Exception as e:
            print("Exception :", str(e))

        return chat_response.text


    #############################################
    def get_response_2(self, query, search_results):

        context = ""
        qdrant_id = search_results[0].id
        print(qdrant_id)


        search_results = qdrant_client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=list(range(max(2, qdrant_id-2), qdrant_id+3))
        )

        print(list(range(max(2, qdrant_id-2), qdrant_id+3)))
        
        for result in search_results:
            context += result.payload['text'] + "\n"

        try:
            chat = self._model.start_chat()
            chat_response = chat.send_message(
                {
                    "parts": [
                        {"text": self._system_prompt},
                        {"text": context},
                        {"text": query}
                    ]
                }
            )
        except Exception as e:
            print("Exception :", str(e))

        return chat_response.text
    

