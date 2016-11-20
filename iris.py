from base import app
from sklearn.externals import joblib

from base import AbstractModel, predict_view


class IrisModel(AbstractModel):
    pipeline = joblib.load('models/iris.pickle')
    fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

models = {'iris': IrisModel()}

# register models
for model_id, model in models.items():
    app.add_url_rule('/{}/predict/'.format(model_id), model_id,
                     predict_view(model), methods=['POST'])
