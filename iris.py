from sklearn.externals import joblib

from base import KeyValueModel, create_app


class IrisModel(KeyValueModel):
    name = 'iris'
    probability = True
    pipeline = joblib.load('models/iris.pickle')
    schema = [
        {'name': 'sepal_length'},
        {'name': 'sepal_width'},
        {'name': 'petal_length'},
        {'name': 'petal_width'}
    ]
    # fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


app = create_app([IrisModel()])
