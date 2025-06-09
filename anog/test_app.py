import pytest
from app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_hello(client):
    response = client.post("/infer", 
                           json={
                                "features": ["S2000",19,"Female",3.8,1.7,3.8,"No",90.2,5.3,"Good",2,"High School","Average",4,"No",58.1],
                                })
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"message": "Hello Alessandro!"}