from flask import Flask
from controller.handler import Handler    

app = Flask(__name__)


@app.route("/")
def home():
    return Handler.HomeRoute()    


if __name__ == "__main__":
    app.run(debug=True)
