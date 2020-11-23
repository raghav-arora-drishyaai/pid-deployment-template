import json
import os
import numpy as np
import pickle
from sklearn.linear_model import Ridge
# from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from utils import mylib



def init():
    
    global model
    model_name = 'diabetes-model'
    model_version = '16'
    ridge_file = 'ridge_0.95.pkl' 
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), ridge_file)
    # model_path = Model.get_model_path('diabetes-model')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # model = pickle.load(model_path)
    # For demonstration purposes only
    # print(mylib.get_alphas())

input_sample = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
output_sample = np.array([3726.995])

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
