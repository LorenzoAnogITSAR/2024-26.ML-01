from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    age = data.get('age')
    gender = data.get('gender')
    study_hours_per_day = data.get('study_hours_per_day')
    social_media_hours = data.get('social_media_hours')
    netflix_hours = data.get('netflix_hours')
    part_time_job = data.get('part_time_job')
    attendance_percentage = data.get('attendance_percentage')
    sleep_hours = data.get('sleep_hours')
    diet_quality = data.get('diet_quality')
    exercise_frequency = data.get('exercise_frequency')
    parental_education_level = data.get('parental_education_level')
    internet_quality = data.get('internet_quality')
    mental_health_rating = data.get('mental_health_rating')
    extracurricular_participation = data.get('extracurricular_participation')
    exam_score = data.get('exam_score')

    mymodel = joblib.load("C:\Users\LorenzoAnog\Downloads\modello-ml\modello_regressione.joblib")
    infer_result = mymodel.predict(exam_score)

    response_data = {
        "result": {
            "value": infer_result
        }
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)