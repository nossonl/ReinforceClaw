"""Slack hook. Adds Good/Bad buttons to every bot reply."""
# wrap(app, model) — drop into any slack_bolt App, done.
# patches app.client.chat_postMessage to append Block Kit buttons.
# button clicks record to nudge db. source="slack".

from nudge import db

_pending = {}  # channel → last user message
_pairs = {}    # ts → {model, prompt, response}

_ACTIONS = [
    {"type": "button", "text": {"type": "plain_text", "text": "Good"}, "action_id": "nudge_good", "style": "primary"},
    {"type": "button", "text": {"type": "plain_text", "text": "Bad"}, "action_id": "nudge_bad", "style": "danger"},
]


def wrap(app, model="unknown"):
    """Patches chat_postMessage to add Good/Bad buttons."""

    @app.event("message")
    def _track(event):
        if not event.get("bot_id"):  # only want human messages
            _pending[event["channel"]] = event.get("text", "(slack)")

    _orig = app.client.chat_postMessage

    def _patched(*, channel, text="", **kw):
        kw["blocks"] = [
            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
            {"type": "actions", "block_id": "nudge_rate", "elements": _ACTIONS},
        ]
        resp = _orig(channel=channel, text=text, **kw)
        _pairs[resp["ts"]] = {"model": model, "prompt": _pending.get(channel, "(slack)"),
                              "response": text}
        return resp

    app.client.chat_postMessage = _patched

    def _tap(body, client, rating):
        ts, ch = body["message"]["ts"], body["channel"]["id"]
        pair = _pairs.pop(ts, None)
        if pair:
            conn = db.connect()
            db.add_feedback(conn, pair["model"], pair["prompt"], pair["response"], rating, source="slack")
            conn.close()
        client.chat_update(channel=ch, ts=ts, text=body["message"].get("text", ""), blocks=[])

    @app.action("nudge_good")
    def _good(ack, body, client): ack(); _tap(body, client, 1)

    @app.action("nudge_bad")
    def _bad(ack, body, client): ack(); _tap(body, client, -1)
