"""Telegram transport and durable processing. Run one instance per data directory."""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import html
import json
import logging
import os
from pathlib import Path
import re
import signal
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

from assistant import Assistant, Decision, render_quote
from pricing import calculate_quote, order_fingerprint, validate_order
from store import Store

ROOT = Path(__file__).resolve().parent
CONTACT = '+79130688284'
log = logging.getLogger('mayker')


@dataclass
class Config:
    bot_token: str
    groq_key: str
    owner_id: int
    webhook_url: str = ''
    webhook_secret: str = ''
    port: int = 8080
    data_dir: Path = ROOT / 'data'
    concurrency: int = 4
    discount_ttl: int = 86400
    context_timeout: int = 7200
    retention_days: int = 30
    model: str = 'openai/gpt-oss-120b'

    @classmethod
    def from_env(cls):
        required = ['BOT_TOKEN', 'GROQ_API_KEY', 'OWNER_CHAT_ID']
        missing = [name for name in required if not os.getenv(name, '').strip()]
        if missing:
            raise ValueError('Не заданы: ' + ', '.join(missing))
        cfg = cls(
            bot_token=os.environ['BOT_TOKEN'].strip(),
            groq_key=os.environ['GROQ_API_KEY'].strip(),
            owner_id=int(os.environ['OWNER_CHAT_ID']),
            webhook_url=os.getenv('WEBHOOK_URL', '').rstrip('/'),
            webhook_secret=os.getenv('WEBHOOK_SECRET', ''),
            port=int(os.getenv('PORT', '8080')),
            data_dir=Path(os.getenv('DATA_DIR', str(ROOT / 'data'))).resolve(),
            concurrency=int(os.getenv('MAX_CONCURRENT_CHATS', '4')),
            discount_ttl=int(os.getenv('DISCOUNT_TTL_SECONDS', '86400')),
            context_timeout=int(os.getenv('CONTEXT_TIMEOUT', '7200')),
            retention_days=int(os.getenv('RETENTION_DAYS', '30')),
            model=os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b'),
        )
        if cfg.owner_id <= 0 or not 1 <= cfg.port <= 65535:
            raise ValueError('OWNER_CHAT_ID должен быть ID личного чата; проверьте PORT.')
        if not 1 <= cfg.concurrency <= 32:
            raise ValueError('MAX_CONCURRENT_CHATS: от 1 до 32.')
        if min(cfg.discount_ttl, cfg.context_timeout, cfg.retention_days) <= 0:
            raise ValueError('Сроки хранения и действия должны быть положительными.')
        if cfg.webhook_url:
            parsed = urlparse(cfg.webhook_url)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.query or parsed.fragment:
                raise ValueError('WEBHOOK_URL должен быть HTTPS URL без query/fragment.')
            if not re.fullmatch(r'[A-Za-z0-9_-]{32,256}', cfg.webhook_secret):
                raise ValueError('WEBHOOK_SECRET: 32–256 символов A–Z, a–z, 0–9, _ или -.')
        for filename in ('prompt.txt', 'catalog.json'):
            if not (ROOT / filename).is_file() or not (ROOT / filename).read_text(encoding='utf-8').strip():
                raise ValueError(f'Обязательный файл {filename} отсутствует или пуст.')
        return cfg


class DeliveryError(Exception):
    def __init__(self, *, permanent=False, retry_after=0):
        # Never include URLs, bot tokens, or Telegram response bodies in exceptions.
        super().__init__('Telegram delivery failed')
        self.permanent = permanent
        self.retry_after = retry_after


class Telegram:
    def __init__(self, client, token):
        self.client = client
        self.base = f'https://api.telegram.org/bot{token}'

    async def call(self, method, payload):
        try:
            response = await self.client.post(f'{self.base}/{method}', json=payload)
            data = response.json()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise DeliveryError() from None
        if not isinstance(data, dict):
            raise DeliveryError()
        if response.status_code == 200 and data.get('ok') is True:
            return data.get('result')
        code = data.get('error_code', response.status_code)
        delay = data.get('parameters', {}).get('retry_after', 0)
        raise DeliveryError(permanent=code in (400, 401, 403, 404),
                            retry_after=delay if isinstance(delay, (int, float)) else 0)


class Groq:
    def __init__(self, client, key, model):
        self.client, self.key, self.model = client, key, model

    async def __call__(self, messages):
        for attempt in range(3):
            delay = 2 ** attempt
            try:
                response = await self.client.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers={'Authorization': f'Bearer {self.key}'},
                    json={'model': self.model, 'messages': messages, 'temperature': 0,
                          'max_tokens': 1500, 'response_format': {'type': 'json_object'}},
                )
                if response.status_code == 200:
                    value = json.loads(response.json()['choices'][0]['message']['content'])
                    if not isinstance(value, dict):
                        raise ValueError('Invalid structured reply')
                    return value
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
                try:
                    delay = min(max(float(response.headers.get('retry-after', delay)), 1), 30)
                except ValueError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            if attempt < 2:
                await asyncio.sleep(delay)
        raise RuntimeError('Assistant temporarily unavailable')


def text_chunks(text, limit=3500):
    """Split before HTML escaping; Telegram limits decoded text, in UTF-16 units."""
    text = re.sub(r'(?m)^\s{0,3}#{1,6}\s+', '', str(text)).strip()
    if not text:
        text = 'Не удалось подготовить ответ. Свяжитесь со студией: ' + CONTACT
    chunks = []
    while text:
        units, end = 0, 0
        for char in text:
            size = 2 if ord(char) > 0xFFFF else 1
            if units + size > limit:
                break
            units += size
            end += 1
        if end < len(text):
            boundary = max(text.rfind('\n', 0, end), text.rfind(' ', 0, end))
            if boundary > end // 2:
                end = boundary
        chunk, text = text[:end], text[end:].lstrip()
        escaped = html.escape(chunk)
        escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped, flags=re.S)
        escaped = escaped.replace('**', '')
        chunks.append(escaped)
    return chunks


def message_actions(chat_id, text, *, after=None, on_failure=None):
    actions = []
    for index, chunk in enumerate(text_chunks(text)):
        parent = after if index == 0 else (index - 1 if after is None else after + index)
        actions.append({'chat_id': chat_id, 'method': 'sendMessage',
                        'payload': {'chat_id': chat_id, 'text': chunk, 'parse_mode': 'HTML'},
                        'after': parent, 'on_failure': on_failure})
    return actions


def message_chat(update):
    if not isinstance(update, dict) or type(update.get('update_id')) is not int:
        return None
    msg = update.get('message')
    if not isinstance(msg, dict):
        return None
    chat = msg.get('chat', {})
    if not isinstance(chat, dict) or chat.get('type') != 'private':
        return None
    chat_id = chat.get('id')
    if type(chat_id) is not int or chat_id <= 0:
        return None
    return chat_id


class Bot:
    def __init__(self, cfg, store, telegram, assistant):
        self.cfg, self.store, self.telegram, self.assistant = cfg, store, telegram, assistant
        self.locks = {}
        self.semaphore = asyncio.Semaphore(cfg.concurrency)
        self.outbox_semaphore = asyncio.Semaphore(cfg.concurrency)

    @contextlib.asynccontextmanager
    async def locked_chats(self, keys):
        # Count waiters too: removing an unlocked lock with waiters creates a race.
        entries = []
        for key in sorted(set(keys)):
            entry = self.locks.setdefault(key, [asyncio.Lock(), 0])
            entry[1] += 1
            entries.append((key, entry))
        try:
            async with contextlib.AsyncExitStack() as stack:
                for _, entry in entries:
                    await stack.enter_async_context(entry[0])
                yield
        finally:
            for key, entry in entries:
                entry[1] -= 1
                if entry[1] == 0:
                    del self.locks[key]

    def state(self, chat_id):
        state = self.store.get_state(chat_id)
        if time.time() - state.get('last_activity', 0) > self.cfg.context_timeout:
            state = {'history': [], 'order': None, 'discount': None,
                     'last_activity': 0, 'handoff_until': 0}
        return state

    def remember(self, state, role, text):
        state['history'] = (state.get('history', []) + [{'role': role, 'content': text}])[-20:]
        state['last_activity'] = time.time()

    @contextlib.asynccontextmanager
    async def typing(self, chat_id):
        async def pulse():
            while True:
                try:
                    await self.telegram.call('sendChatAction', {'chat_id': chat_id, 'action': 'typing'})
                except DeliveryError:
                    pass
                await asyncio.sleep(4)
        task = asyncio.create_task(pulse())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def handoff_actions(self, chat_id, msg, reason, *, attachment=False, confirmation=True,
                        explanation='', order=None, discount=None):
        fallback = message_actions(chat_id, 'Не удалось доставить сообщение менеджеру. '
                                   f'Пожалуйста, свяжитесь со студией: {CONTACT}.')[0]
        sender = msg.get('from', {})
        name = str(sender.get('first_name', 'Клиент'))
        username = str(sender.get('username', ''))
        identity = f'{name}' + (f' (@{username})' if username else '')
        body = msg.get('text') or msg.get('caption') or '[Файл без подписи]'
        text = (f'Запрос клиента {identity}, ID {chat_id}\nПричина: {reason}\n'
                f'{body}\n\n/reply {chat_id} текст\n/discount {chat_id} процент\n/resume {chat_id}')
        if order:
            try:
                text += '\n\nТекущие параметры:\n' + render_quote(order, discount)
            except ValueError:
                text += '\n\nПараметры пока не позволяют сделать стандартный расчёт.'
        actions = message_actions(self.cfg.owner_id, text, on_failure=fallback)
        # Every dependent step must wait for all previous chunks, not only the first.
        for i in range(1, len(actions)):
            actions[i]['after'] = i - 1
        if attachment:
            actions.append({'chat_id': self.cfg.owner_id, 'method': 'copyMessage',
                            'payload': {'chat_id': self.cfg.owner_id, 'from_chat_id': chat_id,
                                        'message_id': msg['message_id']},
                            'after': len(actions) - 1, 'on_failure': fallback})
        if confirmation:
            prefix = explanation.strip() + '\n\n' if explanation.strip() else ''
            actions += message_actions(chat_id, prefix + 'Ваш запрос отправлен менеджеру. '
                                       'Он ответит здесь, когда будет доступен.', after=len(actions) - 1)
        return actions

    async def process(self, row):
        chat_id, update_id, update = row['chat_id'], row['id'], row['update']
        msg = update['message']
        text = msg.get('text') or msg.get('caption') or ''
        target = None
        if chat_id == self.cfg.owner_id and msg.get('from', {}).get('id') == self.cfg.owner_id:
            match = re.match(r'^/(?:reply|discount|resume)(?:@\w+)?\s+(\d+)(?:\s|$)', text)
            if match:
                target = int(match.group(1))
        lock_ids = sorted({chat_id} | ({target} if target else set()))
        async with self.locked_chats(lock_ids):
            async with self.semaphore:
                try:
                    actions, states = await self.plan(chat_id, msg)
                    self.store.commit(update_id, actions, states)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.warning('Update processing failed; retained for retry')
                    if row['attempts'] >= 2:
                        actions = self.handoff_actions(chat_id, msg, 'Не удалось обработать запрос')
                        self.store.commit(update_id, actions, {chat_id: {
                            'history': [], 'order': None, 'discount': None,
                            'last_activity': time.time(),
                            'handoff_until': time.time() + self.cfg.context_timeout}})
                    else:
                        self.store.fail_update(update_id, retry_delay=2 ** row['attempts'])

    async def plan(self, chat_id, msg):
        text = str(msg.get('text') or msg.get('caption') or '')
        command = text.split(maxsplit=1)[0].split('@')[0] if text.startswith('/') else ''
        state = self.state(chat_id)
        if command in ('/reply', '/discount', '/resume'):
            if chat_id != self.cfg.owner_id or msg.get('from', {}).get('id') != self.cfg.owner_id:
                return message_actions(chat_id, 'Команда доступна только владельцу.'), {}
            return await self.admin(chat_id, command, text)
        if command == '/start':
            return message_actions(chat_id, 'Здравствуйте! Я ИИ-помощник студии Майкер. '
                                   'Помогу с ассортиментом и предварительным расчётом. '
                                   'Напишите, что хотите напечатать, на какой вещи и сколько штук. '
                                   'Чтобы начать новый заказ, отправьте /new.'), {}
        if command == '/new':
            return message_actions(chat_id, 'Начинаем новый расчёт. Какое изделие, тираж и размер принта нужны?'), {
                chat_id: {'history': [], 'order': None, 'discount': None,
                          'last_activity': time.time(), 'handoff_until': 0}}
        if command:
            return message_actions(chat_id, 'Доступны /start и /new. Вопрос можно написать обычным сообщением.'), {}
        self.remember(state, 'user', text or '[Клиент отправил файл]')
        attachment = any(key in msg for key in ('document', 'photo', 'video', 'audio', 'voice', 'animation', 'sticker'))
        if attachment or state.get('handoff_until', 0) > time.time():
            if text:
                # Human-mode messages may change quantity/configuration. Never approve a stale quote.
                state['order'] = None
                state['discount'] = None
            state['handoff_until'] = time.time() + self.cfg.context_timeout
            actions = self.handoff_actions(chat_id, msg, 'Макет или вложение' if attachment else 'Продолжение диалога',
                                           attachment=attachment, order=state.get('order'),
                                           discount=state.get('discount'))
            return actions, {chat_id: state}
        if not text:
            return message_actions(chat_id, 'Напишите вопрос текстом или отправьте макет файлом.'), {chat_id: state}
        try:
            async with self.typing(chat_id):
                decision = await self.assistant.respond(text, state['history'][:-1],
                                                        state.get('order'), state.get('discount'))
        except Exception:
            decision = Decision(text='', handoff=True, reason='ИИ-сервис временно недоступен', order={})
        if decision.order is not None:
            state['order'] = decision.order
            discount = state.get('discount')
            if discount and discount.get('order_fingerprint') != order_fingerprint(decision.order):
                state['discount'] = None
        if decision.handoff:
            state['handoff_until'] = time.time() + self.cfg.context_timeout
            actions = self.handoff_actions(chat_id, msg, decision.reason or 'Нужна помощь менеджера',
                                           explanation=decision.text, order=state.get('order'),
                                           discount=state.get('discount'))
        else:
            self.remember(state, 'assistant', decision.text)
            fallback = (message_actions(self.cfg.owner_id,
                        f'Ответ клиенту {chat_id} не доставлен: Telegram отклонил отправку.')[0]
                        if chat_id != self.cfg.owner_id else None)
            actions = message_actions(chat_id, decision.text, on_failure=fallback)
        return actions, {chat_id: state}

    async def admin(self, owner, command, text):
        parts = text.split(maxsplit=2)
        usage = {'/reply': '/reply <ID клиента> текст', '/discount': '/discount <ID клиента> <процент>',
                 '/resume': '/resume <ID клиента>'}[command]
        try:
            target = int(parts[1])
            if target <= 0 or target == owner:
                raise ValueError
        except (IndexError, ValueError):
            return message_actions(owner, 'Формат: ' + usage), {}
        state = self.state(target)
        if command == '/resume':
            state['handoff_until'] = 0
            state['last_activity'] = time.time()
            return message_actions(owner, f'Автоматические ответы для {target} включены.'), {target: state}
        if len(parts) < 3 or not parts[2].strip():
            return message_actions(owner, 'Формат: ' + usage), {}
        if command == '/reply':
            self.remember(state, 'assistant', 'Сообщение менеджера: ' + parts[2])
            state['handoff_until'] = time.time() + self.cfg.context_timeout
            fallback = message_actions(owner, f'Не удалось доставить ответ клиенту {target}.')[0]
            actions = message_actions(target, parts[2], on_failure=fallback)
            for i in range(1, len(actions)):
                actions[i]['after'] = i - 1
            actions += message_actions(owner, f'Сообщение доставлено клиенту {target}.', after=len(actions) - 1)
            return actions, {target: state}
        try:
            percent = Decimal(parts[2].strip().rstrip('%').replace(',', '.'))
            if not percent.is_finite() or not 0 <= percent <= 100:
                raise ValueError
            order = validate_order(state.get('order') or {})
            approval = {'percent': str(percent), 'order_fingerprint': order_fingerprint(order),
                        'expires_at': time.time() + self.cfg.discount_ttl}
            calculate_quote(order, approval)
        except (ValueError, InvalidOperation):
            return message_actions(owner, 'Для скидки нужен заполненный текущий заказ '
                                   '(изделие или свой текстиль, тираж, формат принта) и процент от 0 до 100. '
                                   'Попросите клиента уточнить параметры; /resume вернёт диалог боту.'), {}
        state['discount'] = approval
        state['handoff_until'] = 0
        state['last_activity'] = time.time()
        notification = (f'Менеджер подтвердил скидку {percent}% для текущих параметров заказа. '
                        'При изменении заказа потребуется новое согласование. '
                        '\n\n' + render_quote(order, approval))
        self.remember(state, 'assistant', notification)
        actions = message_actions(target, notification,
                                  on_failure=message_actions(owner, f'Скидка сохранена, но уведомить {target} не удалось.')[0])
        actions += message_actions(owner, f'Скидка {percent}% сохранена для текущего заказа {target}.',
                                   after=len(actions) - 1)
        return actions, {target: state}

    async def deliver(self, row):
        async with self.outbox_semaphore:
            try:
                result = await self.telegram.call(row['method'], row['payload'])
            except DeliveryError as error:
                self.store.fail_outbox(row['id'], retry_delay=max(error.retry_after, 2 ** min(row['attempts'], 6)),
                                       permanent=error.permanent, max_attempts=None)
                log.warning('Telegram delivery failed; queue state updated')
            except asyncio.CancelledError:
                raise
            except Exception:
                self.store.fail_outbox(row['id'], retry_delay=5, max_attempts=None)
                log.warning('Unexpected delivery error; retained for retry')
            else:
                message_id = result.get('message_id') if isinstance(result, dict) else None
                self.store.mark_sent(row['id'], message_id=message_id)

    async def polling(self, stop):
        while not stop.is_set():
            try:
                offset = int(self.store.get_meta('polling_offset', 0))
                updates = await self.telegram.call('getUpdates', {'offset': offset, 'timeout': 20,
                                                                 'allowed_updates': ['message']})
                if not isinstance(updates, list):
                    raise DeliveryError()
                entries = [(u, message_chat(u)) for u in updates if message_chat(u) is not None]
                next_offset = max([offset] + [u['update_id'] + 1 for u in updates
                                             if isinstance(u, dict) and type(u.get('update_id')) is int])
                self.store.ingest_batch(entries, next_offset)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning('Polling unavailable; offset remains durable')
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=5)

    async def workers(self, stop):
        processing, delivering = {}, {}
        last_cleanup = 0
        try:
            while not stop.is_set():
                for active in (processing, delivering):
                    for key, task in list(active.items()):
                        if task.done():
                            with contextlib.suppress(Exception):
                                task.result()
                            del active[key]
                for row in self.store.pending_updates(limit=self.cfg.concurrency * 4):
                    if len(processing) >= self.cfg.concurrency:
                        break
                    if row['chat_id'] not in processing:
                        processing[row['chat_id']] = asyncio.create_task(self.process(row))
                for row in self.store.pending_outbox(limit=self.cfg.concurrency * 4):
                    if len(delivering) >= self.cfg.concurrency:
                        break
                    if row['chat_id'] not in delivering:
                        delivering[row['chat_id']] = asyncio.create_task(self.deliver(row))
                if time.time() - last_cleanup > 3600:
                    self.store.cleanup(retention_days=self.cfg.retention_days)
                    last_cleanup = time.time()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=0.2)
        finally:
            tasks = list(processing.values()) + list(delivering.values())
            if tasks:
                done, pending = await asyncio.wait(tasks, timeout=30)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


def create_app(bot):
    from aiohttp import web

    async def webhook(request):
        supplied = request.headers.get('X-Telegram-Bot-Api-Secret-Token', '')
        if not bot.cfg.webhook_secret or not hmac.compare_digest(supplied.encode(), bot.cfg.webhook_secret.encode()):
            return web.Response(status=403)
        try:
            update = await request.json()
        except Exception:
            return web.Response(status=400)
        chat_id = message_chat(update)
        if chat_id is None:
            return web.Response(status=200)
        try:
            bot.store.ingest(update, chat_id)
        except Exception:
            log.error('Unable to persist webhook update')
            return web.Response(status=503)
        return web.Response(status=200)

    async def health(request):
        return web.Response(text='OK')

    app = web.Application(client_max_size=1024 * 1024)
    app.router.add_post('/webhook', webhook)
    app.router.add_get('/health', health)
    app.router.add_get('/', health)
    return app


async def run(cfg):
    import httpx
    cfg.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(cfg.data_dir, 0o700)
    store = Store(cfg.data_dir / 'bot.sqlite3', context_timeout=cfg.context_timeout)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    runner, poller = None, None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(35, connect=10)) as client:
            telegram = Telegram(client, cfg.bot_token)
            assistant = Assistant(Groq(client, cfg.groq_key, cfg.model), prompt_path=ROOT / 'prompt.txt')
            bot = Bot(cfg, store, telegram, assistant)
            try:
                if cfg.webhook_url:
                    from aiohttp import web
                    runner = web.AppRunner(create_app(bot), access_log=None)
                    await runner.setup()
                    await web.TCPSite(runner, '0.0.0.0', cfg.port).start()
                    await telegram.call('setWebhook', {'url': cfg.webhook_url + '/webhook',
                                                       'secret_token': cfg.webhook_secret,
                                                       'max_connections': 1,
                                                       'allowed_updates': ['message']})
                    log.info('Started in webhook mode')
                else:
                    await telegram.call('deleteWebhook', {'drop_pending_updates': False})
                    poller = asyncio.create_task(bot.polling(stop))
                    log.info('Started in polling mode')
                await bot.workers(stop)
            finally:
                stop.set()
                if poller:
                    poller.cancel()
                    await asyncio.gather(poller, return_exceptions=True)
                if runner:
                    await runner.cleanup()
    finally:
        store.close()


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    # httpx INFO logs otherwise contain Telegram URLs with BOT_TOKEN.
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    try:
        cfg = Config.from_env()
        asyncio.run(run(cfg))
    except (ValueError, FileNotFoundError) as error:
        log.error('Ошибка настройки: %s', error)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        pass
    except Exception:
        log.error('Бот остановлен из-за ошибки запуска или хранилища; проверьте настройки и доступность сервисов.')
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
