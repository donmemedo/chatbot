"""_summary_
"""
import datetime
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from khayyam import JalaliDatetime as jd
import asyncio
from src.config import settings
from dataclasses import dataclass
# from routers.subuser import subuser
from src.logger import logger
import time
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch


def background_loader():
    for model_name_or_path in [
        "makhataei/qa-fa-mdeberta-v3-base",
        "makhataei/qa-persian-albert-fa-zwnj-base-v2",
        "makhataei/qa-persian-xlm-roberta-large",
        "makhataei/qa-persian-mdeberta-v3-base-squad2",
        "makhataei/qa-persian-bert-fa-zwnj-base",
        "makhataei/qa-persian-xlmr-large",
        "makhataei/qa-persian-distilbert-fa-zwnj-base",
        "makhataei/qa-persian-bert-fa-base-uncased",
    ]:
        AutoTokenizer.from_pretrained(model_name_or_path)
        AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    logger.info("Models are loaded in Background.")


def loaders():
    mtokenizer = AutoTokenizer.from_pretrained(settings.MODEL)
    mmodel = AutoModelForQuestionAnswering.from_pretrained(settings.MODEL)
    nlp_cpu = pipeline("question-answering", model=mmodel, tokenizer=mtokenizer, device='cpu')
    try:
        nlp_gpu = pipeline("question-answering", model=mmodel, tokenizer=mtokenizer, device='cuda')
    except:
        logger.critical("There is No GPU")
        nlp_gpu = pipeline("question-answering", model=mmodel, tokenizer=mtokenizer, device='cpu')
    return mtokenizer, mmodel, nlp_cpu, nlp_gpu


background_loader()
loaders()