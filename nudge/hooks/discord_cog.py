"""Discord cog. Adds Good/Bad buttons to every bot reply."""
# drop-in: load this cog into any discord.py bot.
# every bot message gets rating buttons. clicks go to nudge's sqlite db.
# source="discord" so you can filter by platform.

from nudge import db

try:
    import discord
    from discord.ext import commands
except ImportError:
    discord = commands = None


if discord is not None:
    class FeedbackView(discord.ui.View):
        def __init__(self, model, prompt, response):
            super().__init__(timeout=3600)
            self.model, self.prompt, self.response = model, prompt, response

        def _record(self, rating):
            conn = db.connect()
            db.add_feedback(conn, self.model, self.prompt, self.response, rating, source="discord")
            conn.close()

        @discord.ui.button(label="Good", style=discord.ButtonStyle.green)
        async def good(self, interaction, _btn):
            self._record(1)
            await interaction.response.send_message("Rated **good**", ephemeral=True)
            self.stop()

        @discord.ui.button(label="Bad", style=discord.ButtonStyle.red)
        async def bad(self, interaction, _btn):
            self._record(-1)
            await interaction.response.send_message("Rated **bad**", ephemeral=True)
            self.stop()


    class FeedbackCog(commands.Cog):
        def __init__(self, bot, model):
            self.bot, self.model = bot, model
            self._pending = {}  # channel_id → last user message

        @commands.Cog.listener()
        async def on_message(self, msg):
            if msg.author == self.bot.user:
                prompt = self._pending.pop(msg.channel.id, "(discord)")
                view = FeedbackView(self.model, prompt, msg.content)
                await msg.edit(view=view)
            elif not msg.author.bot:
                self._pending[msg.channel.id] = msg.content


    async def setup(bot, model="unknown"):
        """Call this once: await setup(bot, model="your-model")"""
        await bot.add_cog(FeedbackCog(bot, model))
else:
    async def setup(bot, model="unknown"):
        raise ImportError("discord.py is required for nudge.hooks.discord_cog")
