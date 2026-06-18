import aiohttp
from config import Config

class TelegramNotifier:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = (
            self.token and 
            self.chat_id and 
            self.token != "your_telegram_token_here" and 
            self.chat_id != "your_telegram_chat_id_here"
        )
        if self.enabled:
            print("[ALERTS] Telegram notifier enabled.")
        else:
            print("[ALERTS] Telegram alerts disabled (Using default placeholders or empty settings).")

    async def send_message(self, message):
        """
        Sends an alert message to Telegram channel/chat.
        """
        # Always print to console safely
        import sys
        encoding = sys.stdout.encoding or 'utf-8'
        try:
            print(f"[ALERT] {message}")
        except UnicodeEncodeError:
            safe_msg = message.encode(encoding, errors='replace').decode(encoding)
            print(f"[ALERT] {safe_msg}")
        
        if not self.enabled:
            return
            
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": f"🔔 *PrimeSignal Alert*\n\n{message}",
            "parse_mode": "Markdown"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=5) as response:
                    if response.status != 200:
                        text = await response.text()
                        print(f"ERROR: Failed to send Telegram alert: {text}")
        except Exception as e:
            print(f"WARNING: Exception sending Telegram alert: {e}")
