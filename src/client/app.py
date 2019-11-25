import os

from flask import Flask, render_template, request

messages = []


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY="dev")

    if test_config is None:
        app.config.from_pyfile("config.py", silent=True)
    else:
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route("/", methods=["GET", "POST"])
    def epsilon():
        if request.method == "POST":
            messages.append(dict(sender="me", text=request.form["message"]))
            messages.append(
                dict(sender="them", text=request.form["message"][::-1])
            )
        return render_template("epsilon.html", messages=messages)

    return app


if __name__ == "__main__":
    create_app()
