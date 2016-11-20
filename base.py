import numpy as np
from flask import Flask, request
from werkzeug.exceptions import BadRequest

app = Flask(__name__)


class AbstractModel(object):
    pipeline = None
    fields = []

    def extract_variables(self, input):
        """
        :param input: dict-like input
        :return: list of variables values in the order they should be fed
        to the model.
        """
        # a plain "values" field takes precedence
        if 'values' in input.keys():
            values = input.get('values')
            return values.split(',')
        # if that is not found, check that all needed fields are present
        elif (field in input.keys() for field in self.fields):
            return [input[field] for field in self.fields]

    def form_to_array(self, input):
        """
        Transform form input to numpy array.

        Raise ValueError if it fails.

        :param input: dict-like input containing either a "values"
        field or the fields specified in self.fields.
        :return: 2d-array to be fed to a sklearn model.
        """
        vars = self.extract_variables(input)
        if vars:
            # raises ValueError if values are not floats
            vars_float = [float(var) for var in vars]
            return np.array(vars_float).reshape(1, -1)
        # otherwise raise a ValueError
        else:
            raise ValueError(
                'Required fields {} not found in input and "values" was not '
                'found either.'.format(self.fields))

    def predict(self, x):
        return self.pipeline.predict(x)


def predict_view(model):
    def func():
        input = request.form
        try:
            x = model.form_to_array(input)
        except ValueError as error:
            raise BadRequest('Invalid input: {}.'.format(error))

        # perform prediction
        prediction = model.predict(x)

        # return first element
        return str(prediction[0])
    return func
