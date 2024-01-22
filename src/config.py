"""_summary_

Returns:
    _type_: _description_
"""
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModel,
)
import pandas as pd


dd = pd.read_csv("tot.csv")

class Settings(BaseSettings):
    """_summary_

    Args:
        BaseSettings (_type_): _description_
    """

    API_PREFIX: str = ""
    DOCS_URL: str = ""

    OPENAPI_URL: str = ""
    ORIGINS: str = "*"
    ROOT_PATH: str = ""
    SWAGGER_TITLE: str = "ChatBot"
    VERSION: str = "1.0.0"

    APPLICATION_ID: str = "d7f48c21-2a19-4bdb-ace8-48928bff0eb5"
    # GRPC_IP: str = "172.24.65.20"
    # GRPC_PORT: int = 9035
    SPLUNK_HOST: str = "127.0.0.1"
    SPLUNK_PORT: int = 4521
    SPLUNK_INDEX: str = "dev"

    DATE_STRING: str = "%Y-%m-%d"
    FASTAPI_DOCS: str = "/docs"
    FASTAPI_REDOC: str = "/redoc"
    MODEL: str = "makhataei/qa-fa-mdeberta-v3-base"
    ABSTRACT: list = dd["question"].values.tolist()
    ANSWERS: list = dd["answer"].values.tolist()
    LOG_LOCATION: str = "../test/logger.log"
    JSON_LOCATION: str = "../test/logger.json"


def loaders():
    for model_name_or_path in [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ]:
        AutoTokenizer.from_pretrained(model_name_or_path)
        AutoModel.from_pretrained(model_name_or_path)

    model1_cpu = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cpu")
    model2_cpu = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

    sentences = settings.ABSTRACT

    embeddings1_cpu = model1_cpu.encode(sentences)
    embeddings2_cpu = model2_cpu.encode(sentences)
    return model1_cpu,embeddings1_cpu,model2_cpu,embeddings2_cpu



settings = Settings()
DATAS = loaders()