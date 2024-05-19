from flask import Flask, render_template, send_file
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import psycopg2
import os
from datetime import datetime, timedelta
from work import do_work, generate_map
import tempfile

app = Flask(__name__)


def load_insights():
    # Connect to the PostgreSQL database using psycopg2
    DATABASE_URL = os.environ["DATABASE_URL"]
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

    return insights


def schedule_tasks():
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
    # Call load_insights to ensure the most recent data is displayed
    insights = load_insights()
    return render_template("home.html", insights=insights)


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


@app.route("/map")
def map_view():
    # Generate the map and get the path to the temporary file
    map_file_path = generate_map(
        "processed_data/biden_win_probabilities.csv",
        "raw_data/cb_2023_us_state_500k.shp",
    )

    # Serve the map image directly from the temporary file path
    response = send_file(map_file_path, mimetype="image/png")

    # Clean up the temporary file after sending the file
    os.unlink(map_file_path)

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
