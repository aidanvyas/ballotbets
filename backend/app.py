from flask import Flask, jsonify

app = Flask(__name__)

# Placeholder for election forecast data
# In a real-world scenario, this would be replaced with a database call or some other dynamic data source
election_forecast_data = {
    "states": [
        {"name": "Alabama", "electoralVotes": 9, "chanceToWin": 0.6},
        {"name": "Alaska", "electoralVotes": 3, "chanceToWin": 0.4},
        # ... more states
    ],
    "national": {"totalElectoralVotes": 538, "chanceToWin": 0.5}
}

@app.route('/')
def index():
    return 'Election Forecast Application'

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    return jsonify(election_forecast_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
