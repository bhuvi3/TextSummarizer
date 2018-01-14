from flask import Flask, render_template, g, request
from document_generator import *
from lsa_summary import *

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def show():
	inp = request.form["text"]
	#summary = get_summary(inp, 10)
	summary = get_lsa_summary(inp, 10)
	return render_template('index.html', text=summary)

if __name__ == '__main__':
    app.run()
