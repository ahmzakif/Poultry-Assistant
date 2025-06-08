from __future__ import annotations

import logging
from typing import List

from flask import Flask, render_template, request
from werkzeug.wrappers import Response

from chat import ChatService
from database import load_vector_store

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

app = Flask(__name__)

vector_store = load_vector_store()
chat_service = ChatService()

_history_by_ip: dict[str, List[str]] = {}


@app.route("/")
def index() -> str:
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def get_message() -> Response | str:
    """Terima pesan user & kembalikan balasan bot."""
    user_msg: str = request.form["msg"]
    ip = request.remote_addr or "anonymous"
    history = _history_by_ip.setdefault(ip, [])

    history.append(f"User: {user_msg}")
    bot_reply = chat_service.reply(user_msg, history)
    history.append(f"Bot: {bot_reply}")

    return bot_reply


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
