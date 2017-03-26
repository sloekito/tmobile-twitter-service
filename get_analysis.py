from flask import Flask
from flask_restful import Resource, Api
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)


class Score(Resource):
    def get(self):
        # return {'hello': 'world'}
        parser = reqparse.RequestParser()
        parser.add_argument('twitterhandle', type=str)
        return parser.parse_args()
        args = parser.parse_args()

        # if 'twitterhandle' in args:
        th = args['twitterhandle']

        data = {
            'score': 90,
            'twitterhandle': th
        }

        return data


api.add_resource(Score, '/score')

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask
# app = Flask(__name__)
# from flask import request
# from flask import json
# from flask import Response
# from flask_cors import CORS, cross_origin


# CORS(app)

# # cors = CORS(app)
# # app.config['CORS_HEADERS'] = 'application/json'


# @app.route("/twitterstream")
# # @cross_origin() # allow all origins all methods.
# def get_twitter_stream():
#     return "Hello World!"


# @app.route("/score")
# # @cross_origin() # allow all origins all methods.
# def get_score():
#     data = {
#             'twitterhandle': "bla",
#             'score': 90
#         }
#     if 'twitterhandle' in request.args:
#         th = request.args['twitterhandle']

#         data = {
#             'twitterhandle': th,
#             'score': 90
#         }
#         js = json.dumps(data)
#         resp = Response(js, status=200, mimetype='application/json')
#         return resp

#     return data


# if __name__ == "__main__":
#     app.run()
