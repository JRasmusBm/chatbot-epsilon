import os
from random import choice

from flask import Flask, render_template, request

from src.sentiment.sentiment import Sentiment

QUESTION_TURN = "QUESTION"
REVIEW_TURN = "REVIEW"

messages = []
trained_models_folder = "../../trained_models"
turn = QUESTION_TURN


def new_prompt():
    prompts = [
        "Ask me anything!",
    ]
    return choice(prompts)


def answer_question(question):
    return "Blah... blah..."


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY="dev")
    sentiment = Sentiment(
        f"{trained_models_folder}/cnn_10_epochs_given_dataset.pt"
    )

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
    def epsilon():  # pylint: disable=unused-variable
        global messages
        global turn
        if request.method == "POST":
            messages.append(dict(sender="me", text=request.form["message"]))
            if turn == QUESTION_TURN:
                messages.append(
                    dict(
                        sender="them",
                        text=answer_question(request.form["message"]),
                    )
                )
                messages.append(
                    dict(
                        sender="them",
                        text="What do you think about my response?",
                    )
                )
                turn = REVIEW_TURN
            else:
                messages.append(
                    dict(
                        sender="them",
                        text="I am sorry that you feel that way..."
                        if sentiment.eval(request.form["message"]) < 0.5
                        else "That is great to hear!",
                    )
                )
                messages.append(dict(sender="them", text=new_prompt()))
                turn = QUESTION_TURN
        elif request.method == "GET":
            message = new_prompt()
            messages = [dict(sender="them", text=message)]

        return render_template("epsilon.html", messages=messages)

    return app


if __name__ == "__main__":
    create_app()
