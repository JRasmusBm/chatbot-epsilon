import os
from random import choice

from flask import Flask, render_template, request, session

from src.chatbot.chatbot import generate_response, change_sport
from src.sentiment.sentiment import Sentiment

QUESTION_TURN = "QUESTION"
REVIEW_TURN = "REVIEW"
SUBJECT_TURN = "SUBJECT"

trained_models_folder = "../../trained_models"
error_margin = 0.05


def new_prompt():
    prompts = [
        "Ask me something more! Or change sport to talk about by typing \"change subject\"",
    ]
    return choice(prompts)


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(SECRET_KEY="dev")
    sentiment = Sentiment(f"{trained_models_folder}/bert.pt")

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
        if request.method == "POST":
            message = request.form["message"].strip()
            session["messages"].append(dict(sender="me", text=message))
            if session["turn"] == QUESTION_TURN:
                if message.lower() == "change subject":
                    session["messages"].append(dict(sender="them", text="What subject do you want to talk about?"))
                    session["turn"] = SUBJECT_TURN

                else:
                    session["messages"].append(
                        dict(sender="them", text=generate_response(message),)
                    )
                    session["messages"].append(
                        dict(
                            sender="them",
                            text="How do you feel about my response?",
                        )
                    )
                    session["turn"] = REVIEW_TURN
            elif session["turn"] == SUBJECT_TURN:
                session["messages"].append(dict(sender="them", text=change_sport(message.lower()).capitalize()),)
                session["turn"] = QUESTION_TURN
            else:
                score = sentiment.eval(message)
                session["messages"].append(
                    dict(sender="them", text=f"Sentiment Score: {score}")
                )
                if score < 0.5 - error_margin:
                    session["messages"].append(
                        dict(
                            sender="them",
                            text="I am sorry that you feel that way...",
                        )
                    )
                    session["messages"].append(
                        dict(sender="them", text=new_prompt())
                    )
                    session["turn"] = QUESTION_TURN
                elif 0.5 - error_margin <= score <= 0.5 + error_margin:
                    session["messages"].append(
                        dict(
                            sender="them",
                            text="Don't be shy, tell me what you really think!",
                        )
                    )
                else:
                    session["messages"].append(
                        dict(sender="them", text="That is great to hear!",)
                    )
                    session["messages"].append(
                        dict(sender="them", text=new_prompt())
                    )
                    session["turn"] = QUESTION_TURN
        elif request.method == "GET":
            session["messages"] = [dict(sender="them", text= "Hello, I am your friend Epsilon. You can ask me anything about sport! To change subject type \"change subject\" :)")]
            session["turn"] = QUESTION_TURN

        return render_template("epsilon.html", messages=session["messages"])

    return app


if __name__ == "__main__":
    create_app()
