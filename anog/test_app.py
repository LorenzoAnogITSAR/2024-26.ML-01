import pytest
from app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_hello(client):
    response = client.post("/infer", json={"age": 20, 
                                           "gender" : "Male",
                                           "study_hours_per_day": 0.0,
                                           "social_media_hours" : 5.5,
                                           "netflix_hours" : 0.0,
                                           "part_time_job" : "No",
                                           "attendance_percentage" : 97.3,
                                           "sleep_hours" : 6.5,
                                           "diet_quality" : "Fair",
                                           "exercise_frequency" : 2,
                                           "parental_education_level" : "High School",
                                           "internet_quality" : "Good",
                                           "mental_health_rating" : 8,
                                           "extracurricular_participation" : "No",
                                           "exam_score" : 83.3})
    assert response.status_code == 200
    data = response.get_json()
    assert data['result']['value'] == "Hello"