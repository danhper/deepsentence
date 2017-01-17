import json
from os import path

from flask import Flask, render_template, request

from deep_sentence import settings, summarizer

MANIFEST_FILE = path.join(settings.PROJECT_ROOT, 'deep_sentence/webapp/static/manifest.json')

app = Flask(__name__)

with open(path.join(settings.PROJECT_ROOT, 'members.json'), 'r') as f:
    members = json.loads(f.read())

if not app.debug:
    with open(MANIFEST_FILE, 'r') as f:
        manifest = json.loads(f.read())


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


def asset_filename(name):
    if app.debug:
        return name
    else:
        return manifest[name]
app.jinja_env.filters['asset_filename'] = asset_filename
