import numpy as np
from flask import Flask, request
from werkzeug.exceptions import BadRequest


class AbstractModel(object):
    name = None  # override
    pipeline = None  # override

    def form_to_list(self, input):
        """
        Transform given input dict into list of string variables.

        :param input: dict-like input data
        :return: list of string variables
        """
        raise NotImplementedError()

    def to_array(self, vars):
        """
        Convert list of string variables into a 2d numpy array.

        Raises ValueError if any of the variables cannot be
        transformed into a float.

        :param vars: list of string variables
        :return: 2d numpy array to be fed to a sklearn model
        """
        # raises ValueError if values are not floats
        vars_float = [float(var) for var in vars]
        return np.array(vars_float).reshape(1, -1)

    def parse_input(self, input):
        """
        Transform form input to numpy array.

        Raise ValueError upon failure.

        :param input: dict-like input data
        :return: 2d numpy array to be fed to a sklearn model.
        """
        vars = self.form_to_list(input)
        return self.to_array(vars)

    def predict(self, x):
        """
        Return model prediction for given numpy array `x`

        :param x: input numpy array
        :return: prediction numpy array
        """
        return self.pipeline.predict(x)


class ValuesModel(AbstractModel):
    """
    Takes a single parameter "values" of comma-separeted input variables.
    """
    def form_to_list(self, input):
        values = input.get('values')
        if values is None:
            raise ValueError('"values" variable not present in input')
        values_split = values.split(',')
        return [val.strip() for val in values_split]


class KeyValueModel(AbstractModel):
    """
    Takes a list of key-value fields as input variables.

    The ordered list of fields is defined in the 'fields' class attribute.
    """
    fields = []  # to be defined in subclasses

    def form_to_list(self, input):
        try:
            return [input[field] for field in self.fields]
        except KeyError as err:
            raise ValueError(
                'Variable not found in input: {}'.format(err.args[0]))


def predict_view(model):
    """
    Create predict view for given model instance `model`.

    :param model: an instance o a subclass of AbstractModel
    :return: a view that returns model predictions
    """
    def func():
        input = request.form
        try:
            x = model.parse_input(input)
        except ValueError as error:
            raise BadRequest('Invalid input: {}.'.format(error))

        # perform prediction
        prediction = model.predict(x)

        # return first element
        return str(prediction[0])
    return func


def create_app(models):
    """
    Create a Flask app to serve models under /{model_id}/predict/ endpoints

    :param models: AbstractModel models
    :return: Flask app
    """
    app = Flask(__name__)
    # register models
    for model in models:
        app.add_url_rule('/{}/predict/'.format(model.name), model.name,
                         predict_view(model), methods=['POST'])
    return app