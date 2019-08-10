import flask
from flask import request, render_template

from mathsnap.factory import get_snapper

app = flask.Flask(__name__)

math_snapper = get_snapper()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    result = None
    if request.method == 'POST':
        buffer = request.files['image'].read()
        result = math_snapper.process(buffer)
    return render_template('demo.html', result=result)


@app.route('/annotator')
def annotation():
    return render_template('annotator/index.html')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port='80',
        debug=True,
    )
