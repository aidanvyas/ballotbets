from flask import Flask, render_template, send_file
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import psycopg2
import os
from datetime import datetime, timedelta
from work import do_work, generate_map
import tempfile
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_insights():
    """
    Connect to the PostgreSQL database and fetch the latest entry from the work_log table.
    Returns the insights as a dictionary.
    """
    DATABASE_URL = os.environ["DATABASE_URL"]
    insights = {}
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            try:
                # Fetch the latest entry from the work_log table
                cur.execute("SELECT data FROM work_log ORDER BY id DESC LIMIT 1")
                latest_entry = cur.fetchone()

                # Parse the latest entry for insights
                insights = latest_entry[0] if latest_entry else {}
            except Exception as e:
                logging.error(f"Failed to load insights from the database: {e}")
    return insights

def schedule_tasks():
    """
    Execute the do_work function and schedule the next run.
    """
    try:
        do_work()
    finally:
        # Schedule the next run after the current one finishes, with a delay
        scheduler.add_job(
            func=schedule_tasks,
            trigger=DateTrigger(run_date=datetime.now() + timedelta(minutes=1)),
        )

scheduler = BackgroundScheduler()
scheduler.add_job(func=schedule_tasks, trigger=DateTrigger(run_date=datetime.now()))
scheduler.start()

@app.route("/")
def home():
    """
    Home page route. Calls load_insights to ensure the most recent data is displayed.
    """
    insights = load_insights()
    return render_template("home.html", insights=insights)

@app.route("/methodology")
def methodology():
    """
    Methodology page route.
    """
    return render_template("methodology.html")

@app.route("/map")
def map_view():
    """
    Map view route. Serves the generated map image.
    """
    map_file_path = 'static/plots/win_probability_map.png'
    return send_file(map_file_path, mimetype='image/png')

if __name__ == "__main__":
    # Define the port number as a constant
    PORT_NUMBER = 80
    app.run(host="0.0.0.0", port=PORT_NUMBER)
