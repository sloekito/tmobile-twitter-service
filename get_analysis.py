from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin

app = Flask(__name__)
api = Api(app)
#CORS(app)


class Score(Resource):
    def get(self, twitterhandle):
        data = {
            'score': 90,
            'twitterhandle': twitterhandle
        }
        return data

api.add_resource(Score, '/score/<string:twitterhandle>')
        # return {'hello': 'world'}
        # parser = reqparse.RequestParser()
        # parser.add_argument('twitterhandle', type=str)
        # args = parser.parse_args()

        # # if 'twitterhandle' in args:
        # th = args['twitterhandle']

        # data = {
        #     'score': 90,
        #     'twitterhandle': th
        # }
        
        # return data


# api.add_resource(Score, '/score')

if __name__ == '__main__':
    app.run(debug=True)
