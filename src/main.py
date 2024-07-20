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
from src.dornai import dornai

app = FastAPI(
    version=settings.VERSION,
    title=settings.SWAGGER_TITLE,
    docs_url="/docs",
    redoc_url="/redocs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_events():
    logger.info(f"Time of Startup:{jd.now().isoformat()}")
    # tokenizer = AutoTokenizer.from_pretrained(settings.MODEL)
    # model = AutoModelForQuestionAnswering.from_pretrained(settings.MODEL)
    # nlp = pipeline('question-answering', model, tokenizer)
    # # nlp = pipeline('question-answering', model=settings.MODEL, tokenizer=settings.MODEL)
    # return model,tokenizer,nlp
    # loaders()
    logger.info(f"Ready for Your Questions:{jd.now().isoformat()}")


@dataclass
class Question:
    question: str
    # abstract: str = None


@app.post("/qa-test-gpu", tags=["Test-GPU"])
# async def response(question):
async def qa_persian_gpu(qa: Question):
    start = int(1000 * time.time())
    chat = qa.question
    response=dornai(chat)

    result = {
        "Context": chat,  # len(marketers),
        "IsValid": response[0],  # len(marketers),
        "SystemAnswer": response[1],
        # "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
    }
    return JSONResponse(status_code=200, content=result)


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=80)
