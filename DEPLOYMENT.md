# BallotBets Deployment Documentation

## Overview of Changes
The ballotbets site has been updated to enable dynamic updates without the need for manual redeployment. The changes include the integration of a PostgreSQL database for data management and the continuous execution of the `do_work` function to process and update polling data.

## PostgreSQL Database Setup
A PostgreSQL database has been configured to store the work log data. The `db_setup.py` script initializes the database schema, creating a `work_log` table to store the data as JSON.

## Continuous Execution of `do_work` Function
The `do_work` function in `work.py` has been updated to insert data directly into the PostgreSQL database. The Flask application's `main.py` file uses APScheduler's `BackgroundScheduler` to run the `do_work` function continuously, ensuring that the site's data is updated after each execution cycle.

## Flask Application Deployment with Replit Autoscale
The Flask application is configured to run on Replit's autoscale deployments, which requires the server to listen on `0.0.0.0` and expose port `80`. The application is stateless, with data stored in the external PostgreSQL database, adhering to Replit's best practices for autoscale deployments.

## Testing and Validation Process
The dynamic update process has been tested in a controlled environment to ensure that the `do_work` function executes correctly and that the Flask application serves the updated data without errors. The application logs are monitored for any issues, and the PostgreSQL database is checked to confirm that data is being inserted and retrieved as expected.
