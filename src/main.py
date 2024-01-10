"""_summary_
"""
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from khayyam import JalaliDatetime as jd
import asyncio
from src.config import settings, DATAS
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util
from src.logger import logger
import time
from transformers import (
    AutoTokenizer,
    AutoModel,
)


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
    DATAS
    logger.info(f"Ready for Your Questions:{jd.now().isoformat()}")




@dataclass
class Question:
    question: str




@app.post("/response1", tags=["Default"])
async def response1(qa: Question):
    question = qa.question
    embeddings = DATAS[1]
    model = DATAS[0]
    sentences = settings.ABSTRACT
    answ = settings.ANSWERS

    emb1 = model.encode(question)
    cos_sim = util.cos_sim(embeddings, emb1)

    # Add all pairs to a list with their cosine similarity score
    all_sentence_combinations = []
    for i in range(len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i], i])

    # Sort list by the highest cosine similarity score
    all_sentence_combinations = sorted(
        all_sentence_combinations, key=lambda x: x[0], reverse=True
    )

    logger.info("Top-5 most similar pairs:")
    answer = []
    answi = []
    for score, i in all_sentence_combinations[0:5]:
        logger.info(f"{sentences[i]} \t {cos_sim[i]}")
        answer.append(f"{sentences[i]}")

    for score, i in all_sentence_combinations[0:5]:
        logger.info(f"{answ[i]} \t {cos_sim[i]}")
        if score>0.5:
            answi.append(f"{answ[i]}")
    if not answi:
        answi.append("لطفا سوال خود را مشخص‌تر بپرسید یا برای اتصال به اپراتور دکمه زیر را بزنید.")
    logger.info(f"{question} \n {answer}")
    result = {
        "Question": question,  # len(marketers),
        "Answer": answer,  # len(marketers),
        "RAnswer": answi,  # len(marketers),
        # "Context": abstract,
        "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
    }
    return JSONResponse(status_code=200, content=result)


@app.post("/response2", tags=["Default"])
async def response1(qa: Question):
    question = qa.question
    embeddings = DATAS[3]
    model = DATAS[2]
    sentences = settings.ABSTRACT
    answ = settings.ANSWERS

    emb1 = model.encode(question)
    cos_sim = util.cos_sim(embeddings, emb1)

    # Add all pairs to a list with their cosine similarity score
    all_sentence_combinations = []
    for i in range(len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i], i])

    # Sort list by the highest cosine similarity score
    all_sentence_combinations = sorted(
        all_sentence_combinations, key=lambda x: x[0], reverse=True
    )

    logger.info("Top-5 most similar pairs:")
    answer = []
    answi = []
    for score, i in all_sentence_combinations[0:5]:
        logger.info(f"{sentences[i]} \t {cos_sim[i]}")
        answer.append(f"{sentences[i]}")

    for score, i in all_sentence_combinations[0:5]:
        logger.info(f"{answ[i]} \t {cos_sim[i]}")
        if score>0.5:
            answi.append(f"{answ[i]}")
    if not answi:
        answi.append("لطفا سوال خود را مشخص‌تر بپرسید یا برای اتصال به اپراتور دکمه زیر را بزنید.")
    logger.info(f"{question} \n {answer}")
    result = {
        "Question": question,  # len(marketers),
        "Answer": answer,  # len(marketers),
        "RAnswer": answi,  # len(marketers),
        # "Context": abstract,
        "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
    }
    return JSONResponse(status_code=200, content=result)


# @app.post("/response2", tags=["Default"])
# # async def response(question):
# async def response2(qa: Question):
#     question = qa.question
#     embeddings = DATAS[3]
#     model = DATAS[2]
#
#     sentences = settings.ABSTRACT
#     answ = settings.ANSWERS
#     kwargs = {}
#     emb1 = model.encode(question)
#     # Compute cosine similarity between all pairs
#     # cos_sim2 = util.cos_sim(emb1, embeddings)
#     cos_sim = util.cos_sim(embeddings, emb1)
#
#     # Add all pairs to a list with their cosine similarity score
#     all_sentence_combinations = []
#     for i in range(len(cos_sim)):
#         # for j in range(i+1, len(cos_sim2)):#range(i+1, len(cos_sim)):
#         all_sentence_combinations.append([cos_sim[i], i])
#         # all_sentence_combinations.append([cos_sim[i][j], i, j])
#
#     # Sort list by the highest cosine similarity score
#     all_sentence_combinations = sorted(
#         all_sentence_combinations, key=lambda x: x[0], reverse=True
#     )
#
#     logger.info("Top-5 most similar pairs:")
#     # for score, i, j in all_sentence_combinations[0:5]:
#     #     logger.info("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))
#     answer = []
#     answi = []
#     for score, i in all_sentence_combinations[0:5]:
#         # logger.info("{} \t {:.4f}".format(sentences[i], cos_sim[i]))
#         logger.info(f"{sentences[i]} \t {cos_sim[i]}")
#         answer.append(f"{sentences[i]} \t {cos_sim[i]}")
#
#     for score, i in all_sentence_combinations[0:5]:
#         # logger.info("{} \t {:.4f}".format(sentences[i], cos_sim[i]))
#         logger.info(f"{answ[i]} \t {cos_sim[i]}")
#         answi.append(f"{answ[i]} \t {cos_sim[i]}")
#
#     # logger.info(f"{question} {answer}")
#     logger.info(f"{question} \n {answer}")
#     result = {
#         "Question": question,  # len(marketers),
#         "Answer": answer,  # len(marketers),
#         "RAnswer": answi,  # len(marketers),
#         # "Context": abstract,
#         "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
#     }
#     return JSONResponse(status_code=200, content=result)
#

@app.get("/ip-getter", tags=["Default"])
async def read_root(request: Request):
    client_host = request.client.host
    client_scope = request.scope["client"]
    logger.info(f"client host is {client_host}")
    logger.info(f"client scope is {client_scope}")

    return {"client_host": client_host, "client_scope": client_scope}


@app.get("/test", tags=["Default"])
def background_loader():
    for model_name_or_path in [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ]:
        AutoTokenizer.from_pretrained(model_name_or_path)
        AutoModel.from_pretrained(model_name_or_path)
    logger.info("Models are loaded in Background.")

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
    kwargs = {}
    embeddings1_cpu = model1_cpu.encode(sentences)
    embeddings2_cpu = model2_cpu.encode(sentences)
    logger.info("Models are loaded in Background.")
    return model1_cpu,embeddings1_cpu,model2_cpu,embeddings2_cpu


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=80)
