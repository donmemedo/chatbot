import requests
from pymongo import MongoClient

URL1 = "http://37.32.8.187:9001/response1"
URL2 = "http://37.32.8.187:9001/response2"


def get_database():
    connection_sting = "mongodb://37.32.8.187:8081/"
    client = MongoClient(connection_sting)
    database = client["TestAnswerLogs"]

    return database


db = get_database()
db.response1.delete_many({})
db.response2.delete_many({})


def requester(question):
    total_request1 = requests.post(
        URL1,
        json={
            "question": question,
        },
    )

    response1 = total_request1.json()
    try:
        db.response1.insert_one(response1)
    except:
        pass

    total_request2 = requests.post(
        URL2,
        json={
            "question": question,
        },
    )

    response2 = total_request2.json()
    try:
        db.response2.insert_one(response2)
    except:
        pass

    # print(response)


questions = []
for q in questions:
    requester(q)
