# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConfusionDetector is a Flask web application (Python). The project is in its early stages — `app.py` is the main application entry point and `templates/index.html` is the primary HTML template.

## Running the App

```bash
python app.py
```

Or with Flask's dev server:

```bash
flask run
```

## Architecture

- `app.py` — Flask application: routes, request handling, and any ML/detection logic
- `templates/` — Jinja2 HTML templates served by Flask
