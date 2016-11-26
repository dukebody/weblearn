import pytest

from iris import app


@pytest.fixture
def client():
    ctx = app.app_context()
    ctx.push()
    yield app.test_client()

    # teardown
    ctx.pop()
    return


def test_named_values(client):
    data = {
        'sepal_length': 1,
        'sepal_width': 2,
        'petal_length': 3,
        'petal_width': 4
    }
    response = client.post('/iris/predict/', data=data)
    assert response.status_code == 200
    assert response.data == b'2'


def test_named_values_incomplete(client):
    data = {
        'sepal_length': 1,
        'sepal_width': 2,
        'petal_length': 3
    }
    response = client.post('/iris/predict/', data=data)
    assert response.status_code == 400
