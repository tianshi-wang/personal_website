from flask_cors import CORS
from flask import Flask, render_template
import dash
import flask
from pathlib import Path

from webApp.buy2sell import create_buy2sell
from webApp.create_index import create_index

HERE = Path(__file__).parent
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


server = Flask(__name__)
CORS(server)

@server.route('/index')
@server.route('/')
def index():
    return render_template('index.html')
# create_index(app)



@server.route('/about')
def about():
    return render_template('about.html')


@server.route('/contact')
def contact():
    return render_template('contact.html')


@server.route('/post')
def post():
    return render_template('post.html')


app = dash.Dash(__name__, server=server, url_base_pathname='/buy2sell/' )
create_buy2sell(app=app)
# Initialize the dash server and set up the style format



if __name__ == '__main__':
    # Start dash server
    # app = dash.Dash(__name__)
    server.run(host= '0.0.0.0',debug=True)
    print(Path())
    # End of this module
