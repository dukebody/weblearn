import numpy as np
from flask import Flask, request
from werkzeug.exceptions import BadRequest

app = Flask(__name__)


class AbstractModel(object):
    pipeline = None

    def form_to_array(self, values):
        vars = values.split(',')
        # raises ValueError if values are not floats
        vars_float = [float(var) for var in vars]
        return np.array(vars_float)

    def predict(self, x):
        return self.pipeline.predict(x)

    def predict_view(self):
        # post /predict/ values=0,1,2,1,0
        # transform values into (1-n) numpy array --> x
        values = request.form.get('values')

        # fail if field not present
        if values is None:
            raise BadRequest('"values" field not present')

        try:
            x = self.form_to_array(values)
        except ValueError as error:
            raise BadRequest('Invalid input: {}.'.format(error))

        # perform prediction
        prediction = self.predict(x)

        # return first element
        return str(prediction[0])
