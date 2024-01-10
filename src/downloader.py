"""_summary_
"""
from src.logger import logger
from transformers import AutoModel, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util


def background_loader():
    for model_name_or_path in [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ]:
        SentenceTransformer(model_name_or_path)
        AutoTokenizer.from_pretrained(model_name_or_path)
        AutoModel.from_pretrained(model_name_or_path)

    logger.info("Models are loaded in Background.")


background_loader()
