import os
from random import choice

from flask import Flask, render_template, request

from src.chatbot.chatbot import generate_response
from src.sentiment.sentiment import Sentiment

QUESTION_TURN = "QUESTION"
REVIEW_TURN = "REVIEW"

messages = []
trained_models_folder = "../../trained_models"
turn = QUESTION_TURN
error_margin = 0.05


def new_prompt():
    prompts = [
        "Name a tennis-related topic.",
    ]
    return choice(prompts)


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY="dev")
    sentiment = Sentiment(f"{trained_models_folder}/cnn_20_epochs_imdb.pt")

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
            message = request.form["message"].strip()
            messages.append(dict(sender="me", text=message))
            if turn == QUESTION_TURN:
                messages.append(
                    dict(sender="them", text=generate_response(message),)
                )
                messages.append(
                    dict(
                        sender="them",
                        text="How do you feel about my response?",
                    )
                )
                turn = REVIEW_TURN
            else:
                score = sentiment.eval(message)
                messages.append(
                    dict(sender="them", text=f"Sentiment Score: {score}")
                )
                if score < 0.5 - error_margin:
                    messages.append(
                        dict(
                            sender="them",
                            text="I am sorry that you feel that way...",
                        )
                    )
                    messages.append(dict(sender="them", text=new_prompt()))
                    turn = QUESTION_TURN
                elif 0.5 - error_margin <= score <= 0.5 + error_margin:
                    messages.append(
                        dict(
                            sender="them",
                            text="Don't be shy, tell me what you really think!",
                        )
                    )
                else:
                    messages.append(
                        dict(sender="them", text="That is great to hear!",)
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
