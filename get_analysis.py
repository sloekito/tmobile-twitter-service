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

TWITTER_OAUTH_TOKEN = "1733217704-mRoZiblpaY3Dm4PHdusohME83KXfASrnmviJrvS"
TWITTER_OAUTH_SECRET = "QtFUDlVFkaqbrHgSmbJ952OmwmLNymtADdNcSEI7N6iSl"
TWITTER_CONSUMER_KEY = "hkF8zr3wn9obDaRf3RSX0O7qk"
TWITTER_CONSUMER_SECRET = "UqVh1MQkCPdU117MCYm8xeoUJdWlALrZlJ04Sj76ttQOxsrdNz"


@application.route("/score/<id>")
def score(id):
    auth = OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
    auth.set_access_token(TWITTER_OAUTH_TOKEN, TWITTER_OAUTH_SECRET)

    twitterApi = tweepy.API(auth)
    print(id)
    data = {}
    data["id"] = id
    data["score"] = 70

    userInfo = {}

    if id == "a":
        data["id"] = id
        data["score"] = 50

    result = getUserInfo(twitterApi, id)
    userInfo = result
    # print(userInfo)
    data["screen_name"] = userInfo.screen_name
    data["location"] = userInfo.location
    data["description"] = userInfo.description
    data["profile_image_url"] = userInfo.profile_image_url

    # data["description"] = userInfo["description"]
    # data["profile_image_url"] = userInfo["profile_image_url"]
    data["last_tweet"] = userInfo.status.text
    response = application.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )

    # Get last 100 tweets

    tweets = get_tweets(twitterApi, id, 100)

    # Call Travis Api

    # print(tweets)
    return response


def getUserInfo(twitterApi, id):

    # Get the User object for twitter...
    userInfo = twitterApi.get_user(screen_name=id)

    return userInfo

    # with open('tweet_user.json') as data_file:
    #     data = json.load(data_file)
    #     print(data["profile_image_url"])
    #     return data
    # return {}

def get_tweets(twitterApi, id, count):
    statuses = twitterApi.user_timeline(screen_name=id, count=count)
    data = []

    for status in statuses:
        data.append(status.text)

    jsonData = json.dumps(data) 
    #print(jsonData)
    return jsonData

# def getTweets(id):
#     with open('tweet_timeline.json') as data_file:
#         data = json.load(data_file)
#         # print(data)
#         texts = []
#         for item in data:
#             texts.append(item["text"])
#         return texts



if __name__ == "__main__":
    application.run(host='0.0.0.0')
