# WebLearn

This is a proof of concept to create a very simple component to serve
scikit-learn machine learning models over HTTP. The idea is to use this
component as a micro-service inside a more complex infrastructure.

It aims to provide a set of base classes, glue code and code patterns to make
the deployment of these models easy and writing very few lines of code.

It's based on Flask because it's lightweight, easy to use and fairly well
documented.

# Basic usage

There are two base classes for models:

 - `ValuesModel`: Expects a `values` input with all the model numeric
   inputs as comma separated values. Like `values=1,3,7.2`.
 - `KeyValueModel`: Expects a list of key-value fields as input.

The following class-level attributes must be defined:

 - `name`: Id of the model to be used in the API endpoint.
 - `pipeline`: sklearn pipeline that accepts a 2d numpy array as input.
 - `fields`: If using a `KeyValueModel`, a list of names of the POST fields
   expected as input.

Example:

```python
class IrisModel(KeyValueModel):
    name = 'iris'
    pipeline = joblib.load('models/iris.pickle')
    fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
```

In this case, we are creating a model based on the pipeline loaded from a file,
with the name `iris` that takes four numeric inputs.

To serve this model, use the `create_app` function, accepting a list of
model instances:

```python
app = create_app([IrisModel()])
```

This will create a Flask app that serves the given model predictions under the
`/iris/predict/ POST` endpoint. If the input data provided is invalid, it
will return `400 Bad Response`. If it is valid, it will return the model
prediction.

To serve this model, assuming you put your model in a `iris.py` file, run the
following in your shell:

```bash
FLASK_APP=iris.py flask run
```

# Features roadmap

 - Input validation
 - Input transformations
