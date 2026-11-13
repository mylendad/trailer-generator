"""Run the Telegram bot: python -m trailer.bot"""

from __future__ import annotations

import asyncio

from trailer.bot.app import main

if __name__ == "__main__":
    asyncio.run(main())
