from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from proyecto2_deployment import transformar

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Clasificación de género de películas',
    description='Clasificación de género de películas')

ns = api.namespace('predict', 
     description='Predicción géneros de la película')

parser = api.parser()

parser.add_argument(
    'year', 
    type=int, 
    required=True, 
    help='Año del lanzamiento de la película', 
    location='args')

parser.add_argument(
    'title', 
    type=str, 
    required=True, 
    help='Nombre de la película', 
    location='args')

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Trama de la película', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PrediccionApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": transformar(args)  
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)