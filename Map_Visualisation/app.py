from flask import *
app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload():
	return request.values['name']

@app.route('/')
def main_page(name=None):
	return render_template('map.html', name=name)

if __name__ == '__main__':
	app.debug = True #debug on
	app.run()