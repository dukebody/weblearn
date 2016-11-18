from sklearn.externals import joblib

from base import AbstractModel


class IrisModel(AbstractModel):
    pipeline = joblib.load('models/iris.pickle')


models = {'iris': IrisModel()}

# register models
from base import app
for model_id, model in models.items():
    app.add_url_rule('/{}/predict/'.format(model_id), model_id,
                     model.predict_view, methods=['POST'])