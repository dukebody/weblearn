import numpy as np
from flask import Flask, request
from werkzeug.exceptions import BadRequest


class Validator(object):
    def __init__(self, schema):
        self.schema = schema

    def clean_field(self, field, value):
        """
        Transform the given value using the field definition.

        The field definition is of the form:
        {
            'name': field name,
            'default': if present, will be used when no value is provided
                       otherwise ValueError will be raised in this case
            'transform': function accepting a single argument to transform
                         the provided value
        }

        Raise `ValueError` with informative message if there were errors
        cleaning the field.
        """
        if value is None:
            if not 'default' in field:
                raise ValueError('Field value missing and no default set')
            else:
                value = field['default']
        if 'transform' in field:
            try:
                value = field['transform'](value)
            except Exception as err:
                raise ValueError(
                    'Error while transforming: {}'.format(str(err.args)))
        return value

    def validate(self, data):
        """
        Validate and clean the given data.

        If there were errors, return False. The 'errors' dict will contain
        a description of the errors found and 'cleaned_data' will be `None`.

        Otherwise, return `True`. The attribute `cleaned_data` will contain
        a list with the cleaned values of all given fields.
        """
        self.errors = {}
        self.cleaned_data = []
        for field in self.schema:
            name = field['name']
            value = data.get(name)
            try:
                cleaned_value = self.clean_field(field, value)
                self.cleaned_data.append(cleaned_value)
            except ValueError as error:
                self.errors[name] = error
        if self.errors:
            self.cleaned_data = None
            return False
        return True


class AbstractModel(object):
    name = None  # override
    pipeline = None  # override
    probability = False

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

    def predict_proba(self, x):
        """
        Return model probability prediction for given numpy array `x`

        :param x: input numpy array
        :return: prediction numpy array
        """
        return self.pipeline.predict_proba(x)


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
    schema = []  # to be defined in subclasses

    def form_to_list(self, input):
        validator = Validator(self.schema)
        is_valid = validator.validate(input)
        if not is_valid:
            raise ValueError(validator.errors)
        else:
            return validator.cleaned_data


def predict_view(model, predict_method='predict'):
    """
    Create predict view for given model instance `model`.

    :param model: an instance o a subclass of AbstractModel
    :param predict_method: name of the method to invoke in the model to
                           perform the prediction
    :return: a view that returns model predictions
    """
    def func():
        input = request.form
        try:
            x = model.parse_input(input)
        except ValueError as error:
            raise BadRequest('Invalid input: {}.'.format(error))

        # perform prediction
        predict = getattr(model, predict_method)
        prediction = predict(x)

        if len(prediction.shape) == 1:  # single prediction
            return str(prediction[0])
        else:  # predicting multiple class'es
            probas = list(prediction[0])
            return ','.join([str(proba) for proba in probas])
    return func


def create_app(models):
    """
    Create a Flask app to serve models under /{model_id}/predict/ and
    (optionall) /{model_id}/predict_proba/ endpoints.

    :param models: AbstractModel models
    :return: Flask app
    """
    app = Flask(__name__)
    # register models
    for model in models:
        app.add_url_rule('/{}/predict/'.format(model.name),
                         '{}_predict'.format(model.name),
                         predict_view(model), methods=['POST'])
        if model.probability:
            app.add_url_rule(
                '/{}/predict_proba/'.format(model.name),
                '{}_predict_proba'.format(model.name),
                predict_view(model, predict_method='predict_proba'),
                methods=['POST'])
    return app