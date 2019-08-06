import flask
from flask import request, render_template

from mathsnap.factory import get_snapper

app = flask.Flask(__name__)

math_snapper = get_snapper()


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        buffer = request.files['image'].read()
        result = math_snapper.process(buffer)
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port='80',
        debug=True,
    )
