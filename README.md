# ChatBot (FAQ)

The first phase of Chatbots designed for FAQ. It uses [SentenceTransformers ](www.sbert.net), which is a Python framework for state-of-the-art sentence, text and image embeddings.

## 1. Stages
This **Project** has these stages:
### 1.1. Models Loaded on GPU/CPU server:
For being so fast in response, Sample Questions were loaded firstly and then in giving the model and Data for next operations.

### 1.2. Matching Project:
This Stage Finds the most similar **Question** for the ***Inserted Question***. Although, we found the most similar question we check the top 5 similar question for next operations and **Logging** system.

### 1.3. Semantic Answer:
Replays with ***Chosen Answer*** and if there is any need for Operator connects to operator. At last but not least, the confident answer can be chosen by filtering if confidence goes up then the true answer is more possible but risk of ***NoAnswer*** will have been more too.

## 2. Run
```shell
pip install -r requirements.txt
python downloader.py
python main.py

```
After below Message was appeared:
```log
"Ready for Your Questions..."
```
you can start to chat with ***ChatBot***, like this:
```shell
curl -X 'POST' \
'http://HOST:PORT/response' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{ "question": "YOUR QUESTION" }'
```