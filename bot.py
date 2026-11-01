import os
import asyncio
import httpx
import json
import logging
import signal
import time
from datetime import datetime
try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    web = None
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
OWNER_CHAT_ID = 553271460
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "8080"))

MAX_HISTORY_MESSAGES = 10
CONTEXT_TIMEOUT = 7200
RATE_LIMIT_SECONDS = 1.0
TRIGGER_WORDS = ["оплата", "оплатить", "купить", "договор", "карта", "жалоба", "реквизиты", "счёт", "счет"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mayker_bot")

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.txt")


def get_system_prompt() -> str:
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Ты — ИИ-менеджер студии печати Майкер. Отвечай на вопросы клиентов."

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations.log")


def log_conversation(chat_id: int, role: str, text: str, name: str = ""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name_str = f" ({name})" if name else ""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{chat_id}]{name_str} {role}: {text}\n")


chat_history = {}
last_message_time = {}
last_activity_time = {}


def check_rate_limit(user_id: int) -> bool:
    now = time.time()
    last = last_message_time.get(user_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        return False
    last_message_time[user_id] = now
    return True


def get_history(user_id: int) -> list:
    now = time.time()
    if user_id in last_activity_time:
        if now - last_activity_time[user_id] > CONTEXT_TIMEOUT:
            chat_history[user_id] = []
            log.info(f"Context reset for user {user_id} (timeout)")
    last_activity_time[user_id] = now
    return chat_history.get(user_id, [])


async def ask_groq(user_id: int, user_message: str) -> str:
    history = get_history(user_id)
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.7, "max_tokens": 1024},
                )
            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                answer = answer.replace("+791-****-8284", "+79130688284")
                answer = answer.replace("+791****8284", "+79130688284")
                answer = answer.replace("+7 913 *** ** 84", "+79130688284")
                if user_id not in chat_history:
                    chat_history[user_id] = []
                chat_history[user_id].append({"role": "user", "content": user_message})
                chat_history[user_id].append({"role": "assistant", "content": answer})
                if len(chat_history[user_id]) > MAX_HISTORY_MESSAGES:
                    chat_history[user_id] = chat_history[user_id][-MAX_HISTORY_MESSAGES:]
                return answer
            elif response.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                log.error(f"Groq API error: {response.status_code} {response.text[:200]}")
                return f"Ошибка API: {response.status_code}. Попробуй позже."
        except Exception as e:
            log.error(f"Groq request failed: {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
            else:
                return "Сервис временно недоступен. Попробуй позже."

    return "Сервис временно недоступен. Попробуй позже."


async def send_message(chat_id: int, text: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{TG_API}/sendMessage", json={"chat_id": chat_id, "text": text})
        return r.json()


async def process_update(update: dict):
    msg = update.get("message")
    if not msg or not msg.get("text"):
        return

    chat_id = msg["chat"]["id"]
    text = msg["text"]
    user = msg.get("from", {})

    client_name = user.get("first_name", "")
    log_conversation(chat_id, "КЛИЕНТ", text, client_name)

    if not check_rate_limit(chat_id):
        await send_message(chat_id, "Пожалуйста, не спешите — я обрабатываю ваш прошлый запрос.")
        return

    if text.startswith("/reply "):
        if chat_id != OWNER_CHAT_ID:
            await send_message(chat_id, "Команда доступна только владельцу.")
            return
        parts = text.split(" ", 2)
        if len(parts) < 3:
            await send_message(chat_id, "Формат: /reply <ID клиента> текст сообщения")
            return
        try:
            target_id = int(parts[1])
        except ValueError:
            await send_message(chat_id, "ID клиента должен быть числом.")
            return
        reply_text = parts[2]
        try:
            result = await send_message(target_id, reply_text)
            if result.get("ok"):
                await send_message(chat_id, f"Сообщение отправлено клиенту {target_id}.")
            else:
                await send_message(chat_id, f"Ошибка: {result.get('description', 'неизвестная ошибка')}")
        except Exception as e:
            await send_message(chat_id, f"Ошибка отправки: {e}")
        return

    if text.startswith("/start"):
        await send_message(chat_id, "Привет! Я ИИ-менеджер студии Майкер. Помогу с ценами, материалами и заказом печати.\nНапиши свой вопрос — например: «Сколько стоит печать на футболке?»")
        return

    text_lower = text.lower()
    triggered = [w for w in TRIGGER_WORDS if w in text_lower]
    if triggered:
        client_name = user.get("first_name", "Неизвестно")
        client_username = user.get("username", "")
        username_str = f" (@{client_username})" if client_username else ""
        await send_message(OWNER_CHAT_ID, f"Клиент {client_name}{username_str} (ID {chat_id}) пишет о: {', '.join(triggered)}\nСообщение: {text}\n\nОтветить: /reply {chat_id} текст")
        await send_message(chat_id, "Передаю ваш запрос менеджеру, он свяжется с вами в течение 5 минут.")
        return

    await send_message(chat_id, "Думаю...")
    answer = await ask_groq(chat_id, text)
    await send_message(chat_id, answer)
    log_conversation(chat_id, "БОТ", answer)

    if "руководител" in answer.lower() or "индивидуальный" in answer.lower():
        client_name = user.get("first_name", "Неизвестно")
        client_username = user.get("username", "")
        username_str = f" (@{client_username})" if client_username else ""
        await send_message(OWNER_CHAT_ID, f"Крупный заказ!\nКлиент: {client_name}{username_str} (ID {chat_id})\nСообщение: {text}\nОтвет бота: {answer}\n\nОтветить: /reply {chat_id} текст")


if HAS_AIOHTTP:
    async def webhook_handler(request: web.Request) -> web.Response:
        try:
            data = await request.json()
            await process_update(data)
        except Exception as e:
            log.error(f"Webhook error: {e}")
        return web.Response(status=200)

    async def health_handler(request: web.Request) -> web.Response:
        return web.Response(text="OK", status=200)

    app = web.Application()
    app.router.add_post("/webhook", webhook_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)
else:
    app = None
    webhook_handler = None
    health_handler = None


async def setup_webhook():
    if WEBHOOK_URL:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{TG_API}/setWebhook", json={"url": f"{WEBHOOK_URL}/webhook"})
            log.info(f"Webhook set: {r.json()}")


async def main():
    _shutdown = False

    loop = asyncio.get_event_loop()

    def _signal_handler(sig):
        nonlocal _shutdown
        log.info(f"Received signal {sig}, shutting down gracefully...")
        _shutdown = True

    # Graceful shutdown - Windows compatible
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: _signal_handler(s))
            except (NotImplementedError, RuntimeError):
                pass
    except Exception:
        pass

    if WEBHOOK_URL and HAS_AIOHTTP:
        await setup_webhook()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", PORT)
        await site.start()
        log.info(f"Bot started in webhook mode on port {PORT}")
        while not _shutdown:
            await asyncio.sleep(1)
        await runner.cleanup()
        log.info("Shutdown complete.")
    else:
        if WEBHOOK_URL and not HAS_AIOHTTP:
            log.warning("aiohttp not available, falling back to polling mode")
        async with httpx.AsyncClient() as client:
            await client.post(f"{TG_API}/deleteWebhook")
        log.info("Bot started in polling mode")
        offset = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            while not _shutdown:
                try:
                    resp = await client.get(
                        f"{TG_API}/getUpdates",
                        params={"offset": offset, "timeout": 5},
                    )
                    data = resp.json()
                    if not data.get("ok"):
                        log.error(f"TG error: {data}")
                        await asyncio.sleep(5)
                        continue
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1
                        await process_update(update)
                except Exception as e:
                    log.error(f"Polling error: {e}")
                    await asyncio.sleep(5)
        log.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
