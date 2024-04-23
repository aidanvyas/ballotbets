from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Placeholder for election forecast data
# In a real-world scenario, this would be replaced with a database call or some other dynamic data source
election_forecast_data = {
    "states": [
        {"name": "Alabama", "electoralVotes": 9, "chanceToWin": 0.6},
        {"name": "Alaska", "electoralVotes": 3, "chanceToWin": 0.4},
        # ... more states
        # Adding more states for demonstration purposes
        {"name": "Arizona", "electoralVotes": 11, "chanceToWin": 0.5},
        {"name": "Arkansas", "electoralVotes": 6, "chanceToWin": 0.7},
        {"name": "California", "electoralVotes": 55, "chanceToWin": 0.3},
        {"name": "Colorado", "electoralVotes": 9, "chanceToWin": 0.6},
        {"name": "Connecticut", "electoralVotes": 7, "chanceToWin": 0.4},
        {"name": "Delaware", "electoralVotes": 3, "chanceToWin": 0.8},
        {"name": "Florida", "electoralVotes": 29, "chanceToWin": 0.5},
        {"name": "Georgia", "electoralVotes": 16, "chanceToWin": 0.4},
        {"name": "Hawaii", "electoralVotes": 4, "chanceToWin": 0.7},
        # ... remaining states
    ],
    "national": {"totalElectoralVotes": 538, "chanceToWin": 0.5}
}

@app.route('/')
def index():
    return 'Election Forecast Application'

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    response = jsonify(election_forecast_data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
