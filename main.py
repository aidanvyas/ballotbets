from flask import Flask, render_template
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import os
from datetime import datetime, timedelta
from work import do_work
import traceback
import psycopg2  # Ensure correct spelling and appropriate import

app = Flask(__name__)

def load_insights():
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        print("DATABASE_URL is not set.")
        return

    insights = {}
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        cur.execute("SELECT data FROM work_log ORDER BY id DESC LIMIT 1")
        latest_entry = cur.fetchone()

        insights = latest_entry[0] if latest_entry else {}

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Failed to load insights from the database: {e}")
        traceback.print_exc()

    app.config['INSIGHTS'] = insights

def schedule_tasks():
    try:
        do_work()
    finally:
        scheduler.add_job(func=schedule_tasks, trigger=DateTrigger(run_date=datetime.now() + timedelta(minutes=1)))

try:
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=schedule_tasks, trigger=DateTrigger(run_date=datetime.now()))
    scheduler.start()
except Exception as e:
    print(f"Scheduler initialization failed: {e}")
    traceback.print_exc()

@app.route('/')
def home():
    return render_template('home.html', insights=app.config.get('INSIGHTS', {}))

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

if __name__ == '__main__':
    try:
        load_insights()
        app.run(host='0.0.0.0', port=80)
    except Exception as e:
        print(f"Flask failed to start: {e}")
        traceback.print_exc()