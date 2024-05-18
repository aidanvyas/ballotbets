from flask import Flask, render_template
import psycopg2
import os

app = Flask(__name__)

def load_insights():
    # Connect to the PostgreSQL database using psycopg2
    DATABASE_URL = os.environ['DATABASE_URL']
    insights = {}
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Fetch the latest entry from the work_log table
        cur.execute("SELECT data FROM work_log ORDER BY id DESC LIMIT 1")
        latest_entry = cur.fetchone()

        # Parse the latest entry for insights
        insights = latest_entry[0] if latest_entry else {}

        # Close the connection
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Failed to load insights from the database: {e}")

    app.config['INSIGHTS'] = insights

def schedule_tasks():
    try:
        do_work()
    finally:
        # Schedule the next run after the current one finishes
        scheduler.add_job(func=schedule_tasks, trigger=DateTrigger(run_date=datetime.now()))

scheduler = BackgroundScheduler()
scheduler.add_job(func=schedule_tasks, trigger=DateTrigger(run_date=datetime.now()))
scheduler.start()

@app.route('/')
def home():
    return render_template('home.html', insights=app.config.get('INSIGHTS', {}))

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

if __name__ == '__main__':
    load_insights()  # Initial load of insights
    app.run(host='0.0.0.0', port=80)
