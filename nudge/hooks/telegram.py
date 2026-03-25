"""Telegram wrapper. Adds Good/Bad/Skip buttons to every bot reply."""
# one function: wrap(app, model). call it on any python-telegram-bot Application.
# patches send_message to append inline buttons. taps go to nudge's sqlite db.
# source="telegram".

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import CallbackQueryHandler, MessageHandler, filters, Application
from nudge import db

_BUTTONS = InlineKeyboardMarkup([[
    InlineKeyboardButton("Good", callback_data="nudge:1"),
    InlineKeyboardButton("Bad", callback_data="nudge:-1"),
    InlineKeyboardButton("Skip", callback_data="nudge:0"),
]])


def wrap(app: Application, model="unknown"):
    """Monkey-patch the app to add feedback buttons on every reply. One call."""

    async def _track(update: Update, ctx):
        if update.message and update.message.text:
            ctx.bot_data.setdefault("last_user_msg", {})[update.message.chat_id] = update.message.text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _track), group=-1)

    _orig = app.bot.send_message

    async def _patched(chat_id, text, *a, **kw):
        kw["reply_markup"] = _BUTTONS
        msg = await _orig(chat_id, text, *a, **kw)
        prompt = app.bot_data.get("last_user_msg", {}).get(chat_id, "(telegram)")
        app.bot_data.setdefault("pairs", {})[msg.message_id] = {
            "model": model, "prompt": prompt, "response": text}
        return msg
    app.bot.send_message = _patched

    async def _on_tap(update: Update, ctx):
        q = update.callback_query
        if not q.data.startswith("nudge:"):
            return
        rating = int(q.data.split(":")[1])
        pair = ctx.bot_data.get("pairs", {}).pop(q.message.message_id, None)
        if pair and rating != 0:
            conn = db.connect()
            db.add_feedback(conn, pair["model"], pair["prompt"], pair["response"],
                            rating, source="telegram")
            conn.close()
        await q.answer({1: "Good", -1: "Bad", 0: "Skipped"}[rating])
        await q.message.edit_reply_markup(reply_markup=None)

    app.add_handler(CallbackQueryHandler(_on_tap, pattern="^nudge:"))
