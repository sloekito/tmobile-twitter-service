from __future__ import absolute_import, print_function

from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin

from flask import json

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy

application = Flask(__name__)
api = Api(application)
CORS(application)


@application.route("/score/<id>")
def score(id):
    print(id)
    data = {}
    data["id"] = id
    data["score"] = 70

    if id == "a":
        data["id"] = id
        data["score"] = 50

    
    userInfo = getUserInfo(id)
    data["profile_image_url"] = userInfo["profile_image_url"]
    response = application.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

    tweets = getTweets(id)

    print(tweets)
    return response



def getUserInfo(id):
    # userInfo = self.api.user_timeline(screen_name=id, count=100)
    with open('tweet_user.json') as data_file:
        data = json.load(data_file)
        print(data["profile_image_url"])
    
        return data
    return {}

def getTweets(id):
    with open('tweet_timeline.json') as data_file:
        data = json.load(data_file)
        # print(data)
        texts = []
        for item in data:
            texts.append(item["text"])
        return texts

    return {}
    # statuses = self.api.user_timeline(screen_name=id, count=100)
    # for status in statuses:
    #     print(status.text)


if __name__ == "__main__":
    application.run(host='0.0.0.0')
