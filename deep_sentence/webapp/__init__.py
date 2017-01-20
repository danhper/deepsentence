import json
from os import path

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, socketio, join_room

from deep_sentence import settings, summarizer
from deep_sentence.logger import logger

MANIFEST_FILE = path.join(settings.PROJECT_ROOT, 'deep_sentence/webapp/static/manifest.json')

app = Flask(__name__)
socketio = SocketIO(app)


with open(path.join(settings.PROJECT_ROOT, 'members.json'), 'r') as f:
    members = json.loads(f.read())


if not app.debug:
    with open(MANIFEST_FILE, 'r') as f:
        manifest = json.loads(f.read())


if path.isfile(settings.GOOGLE_API_CREDENTIALS):
    import budou
    parser = budou.authenticate(settings.GOOGLE_API_CREDENTIALS)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summary.json')
def get_summary():
    progress_callback = make_progress_callback(request.headers.get('x-client-id'))
    urls = request.args.getlist('urls[]')
    title, summary, error = summarize_urls(urls, progress_callback=progress_callback)
    if error:
        resp = jsonify({'error': error})
        resp.status_code = 500
        return resp
    else:
        return jsonify({'title': title, 'summary': summary})


@app.route('/about')
def about():
    return render_template('about.html', members=members)


@socketio.on('subscribe')
def on_join(data):
    room = data['room']
    join_room(room)


@app.template_filter()
def asset_filename(name):
    if app.debug:
        return name
    else:
        return manifest[name]


@app.template_filter()
def japanese_html(text):
    if not parser:
        return text
    result = parser.parse(text, 'wordwrap')
    return result['html_code']


def make_progress_callback(room):
    if room:
        def callback(progress):
            socketio.emit('progress', data={'progress': progress}, room=room)
        return callback
    else:
        return lambda progress: None

def summarize_urls(urls, progress_callback=lambda _: None):
    if not urls:
        return '', '', ''

    logger.info(urls)

    try:
        title, summary = summarizer.summarize_urls(urls, progress_callback=progress_callback)
        return title, summary, ''
    except BaseException as e:
        logger.exception(e)
        return '', '', str(e)


if __name__ == '__main__':
    socketio.run(app)
