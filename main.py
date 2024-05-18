from flask import Flask, render_template
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from datetime import datetime
from work import do_work
import csv

app = Flask(__name__)

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
    csv_file_path = 'static/work_log.csv'
    insights = {}
    try:
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for index, row in enumerate(reader):
                if index == 0:
                    insights['polling_averages'] = row
                elif index == 1:
                    insights['close_states'] = row
                elif index == 2:
                    insights['electoral_college_votes'] = row
                elif index == 3:
                    insights['simulations_summary'] = row
    except Exception as e:
        print(f"Failed to read insights from CSV: {e}")
    return render_template('home.html', insights=insights)

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
