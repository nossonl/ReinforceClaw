"""Matrix/Element hook. Reaction thumbs-up/down on bot messages → nudge DB."""
# wraps an existing nio.AsyncClient. tracks last user message per room as prompt.
# reacts to m.reaction events on messages we sent. source="matrix".

from nudge import db

_pending = {}  # room_id → last user message
_bot_msgs = {}  # event_id → {model, prompt, response}
_GOOD = {"\U0001f44d", "\U0001f44d\U0001f3fb", "\U0001f44d\U0001f3fc", "\U0001f44d\U0001f3fd", "\U0001f44d\U0001f3fe", "\U0001f44d\U0001f3ff"}
_BAD = {"\U0001f44e", "\U0001f44e\U0001f3fb", "\U0001f44e\U0001f3fc", "\U0001f44e\U0001f3fd", "\U0001f44e\U0001f3fe", "\U0001f44e\U0001f3ff"}


def wrap(client, model="unknown"):
    """Register callbacks on an existing nio.AsyncClient."""
    import nio

    user_id = client.user_id

    async def _on_message(room, event):
        if event.sender == user_id:
            # our own message — track for rating
            _bot_msgs[event.event_id] = {
                "model": model,
                "prompt": _pending.get(room.room_id, "(matrix)"),
                "response": event.body,
            }
        else:
            _pending[room.room_id] = event.body

    async def _on_reaction(room, event):
        if not isinstance(event, nio.UnknownEvent) or event.type != "m.reaction":
            return
        content = event.source.get("content", {})
        rel = content.get("m.relates_to", {})
        target = rel.get("event_id", "")
        emoji = rel.get("key", "")

        pair = _bot_msgs.get(target)
        if not pair:
            return
        if emoji in _GOOD:
            rating = 1
        elif emoji in _BAD:
            rating = -1
        else:
            return
        conn = db.connect()
        db.add_feedback(conn, pair["model"], pair["prompt"], pair["response"], rating, source="matrix")
        conn.close()
        _bot_msgs.pop(target, None)

    client.add_event_callback(_on_message, nio.RoomMessageText)
    client.add_event_callback(_on_reaction, nio.UnknownEvent)
    return client
