import pytest

from iris import app


@pytest.fixture
def client(request):
    ctx = app.app_context()
    ctx.push()
    yield app.test_client()

    # teardown
    ctx.pop()
    return


def test_predict(client):
    response = client.post('/iris/predict/', data={'values': '1,2,3,4'})
    assert response.status_code == 200
    assert response.data == b'2'


def test_no_values(client):
    response = client.post('/iris/predict/', data={})
    assert response.status_code == 400


def test_values_nofloat(client):
    response = client.post('/iris/predict/', data={'values': '1a,2,3,4'})
    assert response.status_code == 400


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
