import json
from os import path

from flask import Flask, render_template, request

from deep_sentence import settings, summarizer

app = Flask(__name__)

with open(path.join(settings.PROJECT_ROOT, 'members.json'), 'r') as f:
    members = json.loads(f.read())

@app.route('/')
def index():
    summary, error = '', None
    urls = request.args.getlist('urls[]')
    if urls:
        app.logger.info(urls)
        try:
            summary = summarizer.summarize_urls(urls)
        except BaseException as e:
            error = str(e)
    return render_template('index.html', summary=summary, error=error)

@app.route('/about')
def about():
    return render_template('about.html', members=members)
