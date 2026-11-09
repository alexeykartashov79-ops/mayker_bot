"""The model extracts intent; local code owns facts, money and authority."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import re

from pricing import (CATALOG, MissingDetails, NeedsManager, approved_percent,
                     calculate_quote, money, validate_order)


@dataclass
class Decision:
    text: str
    handoff: bool = False
    reason: str = ''
    order: dict | None = None


PRIVACY = re.compile(
    r'персональн|конфиденц|приватност|политик[аиу].*данн|'
    r'хран\w*.*данн|данн\w*.*хран|обработ\w*.*данн|данн\w*.*обработ|'
    r'удал\w*.*данн|данн\w*.*удал|сервер|дата.?центр|шифро|aes.?256|152.?фз|gdpr', re.I)
PAYMENT_READY = re.compile(
    r'реквизит|готов\w*\s+(?:\w+\s+){0,3}(?:оплат|переве|плат)|'
    r'хочу\s+оплат|как\s+оплат|куда\s+(?:\w+\s+){0,2}(?:перев|плат)|'
    r'оплатил|оплатила|уже\s+перев[её]л|выстав\w*.*сч[её]т|сч[её]т\s+на\s+оплат', re.I)
COMPLAINT = re.compile(r'жалоб|претензи|верн\w*\s+деньг|возврат|брак|обман', re.I)
HUMAN = re.compile(r'(?:позов|приглас|соедин|свяж|нужен|нужна|хочу\s+поговорить).*'
                   r'(?:менеджер|руководител|человек|алексе)', re.I)
DISCOUNT_CLAIM = re.compile(r'скидк', re.I)
AUTHORITY_CLAIM = re.compile(r'обещ|соглас|одобр|разреш|алексе|руководител|главн', re.I)
PORTFOLIO = re.compile(
    r'портфолио|пример\w*\s+(?:ваших\s+)?работ|ваши\s+работы|'
    r'(?:где\s+(?:можно\s+)?посмотреть|покаж\w*)\s+(?:ваши\s+)?примеры', re.I)


def _product_lines(include_materials=False, tshirts_only=False):
    lines = []
    for item in CATALOG['products'].values():
        if tshirts_only and not item['name'].casefold().startswith('футболк'):
            continue
        price = f"{money(Decimal(item['price']))} ₽/шт." if item['price'] else 'цена по запросу'
        line = f"- {item['name']}: {price}"
        if include_materials:
            line += f"; {item['material']}"
        lines.append(line)
    return '\n'.join(lines)


def _print_prices():
    return '\n'.join(f'- {key}: {value} ₽/нанесение' for key, value in CATALOG['print_prices'].items())


def render_faq(topic: str, order: dict) -> str | None:
    if not isinstance(topic, str):
        return None
    facts = CATALOG['facts']
    if topic in facts:
        return facts[topic]
    if topic == 'catalog':
        return '**Изделия**\n' + _product_lines(True) + '\n\n' + facts['sales'] + '\n\n**Печать**\n' + _print_prices()
    if topic == 'prices':
        return '**Текстиль**\n' + _product_lines() + '\n\n**Печать**\n' + _print_prices() + '\n\n' + facts['sales'] + '\n' + facts['discounts']
    if topic == 'materials':
        product = CATALOG['products'].get(order.get('product'))
        if product:
            return f"{product['name']}: {product['material']}."
        return '\n'.join(f"- {item['name']}: {item['material']}" for item in CATALOG['products'].values())
    if topic == 'sizes':
        keys = [order['product']] if order.get('product') in CATALOG['sizes'] else list(CATALOG['sizes'])
        result = []
        for key in keys:
            result.append('**' + CATALOG['products'][key]['name'] + '**')
            result.extend(f'- {size}: {values}' for size, values in CATALOG['sizes'][key].items())
        result.append('\nЧисла приведены в порядке исходной размерной сетки. Подписи колонок и единицы измерения в ней не указаны — их нужно уточнить у менеджера.')
        return '\n'.join(result)
    if topic == 'contacts':
        s = CATALOG['studio']
        return (f"{s['name']}: {s['phone']} (телефон, Telegram / WhatsApp).\n"
                f"Почта: {s['email']}\nСайт: {s['site']}\n"
                f"Режим работы: {s['hours']}.\nВыдача и приём текстиля: {s['pickup_hours']}.\n"
                f"Адрес: {s['address']}.\nРаботаем онлайн. Самовывоз и передачу текстиля согласуйте с менеджером; консультации в офисе не предусмотрены.")
    return None


def render_quote(order: dict, discount: dict | None = None) -> str:
    quote = calculate_quote(order, discount)
    name = 'Ваш текстиль' if quote.order['own_textile'] else CATALOG['products'][quote.order['product']]['name']
    specifications = [name, f"{quote.quantity} шт.", f"одно нанесение {quote.order['print_size']} на изделие"]
    for key in ('color', 'size'):
        if order.get(key):
            specifications.append(order[key])
    approval = 'согласована руководителем для этого заказа' if quote.approved_discount else 'стандартная'
    lines = ['**Предварительный расчёт**', ', '.join(specifications) + '.',
             f'Скидка: {money(quote.discount_percent)}% ({approval}).',
             f'Текстиль: {money(quote.textile_before)} → {money(quote.textile_after)} ₽.',
             f'Печать: {money(quote.printing_before)} → {money(quote.printing_after)} ₽.']
    if quote.minimum_adjustment:
        lines.append(f"Применён минимум заказа на вашем текстиле: {CATALOG['own_textile_minimum']} ₽.")
    lines.append(f'**Итого: {money(quote.total)} ₽.**')
    lines.append('Без доставки. Для окончательного подтверждения заказа и оплаты нужен менеджер.')
    return '\n'.join(lines)


class Assistant:
    def __init__(self, llm, prompt_path: Path | None = None):
        self.llm = llm
        path = Path(prompt_path) if prompt_path is not None else Path(__file__).with_name('prompt.txt')
        self.prompt = path.read_text(encoding='utf-8').strip()
        if not self.prompt:
            raise ValueError(f'Пустой системный промпт: {path}')

    @staticmethod
    def handoff(order, reason, text='Для этого вопроса нужна помощь менеджера.'):
        # Runtime sends this only with a delivery-dependent handoff confirmation.
        return Decision(text, handoff=True, reason=reason, order=order)

    async def respond(self, text: str, history: list[dict], order: dict | None = None,
                      discount: dict | None = None) -> Decision:
        current = validate_order(order)
        if PRIVACY.search(text):
            return self.handoff({}, 'privacy',
                'У меня нет подтверждённых сведений о сроках хранения, расположении серверов и других условиях обработки персональных данных. Их должен уточнить руководитель студии.')
        if COMPLAINT.search(text):
            return self.handoff({}, 'complaint', 'Ваш вопрос должен разобрать менеджер студии.')
        # A handoff request can also change the order: "now twenty, ready to
        # pay". Extract those changes before freezing the approval target.
        forced_reason = None
        forced_text = 'Для этого вопроса нужна помощь менеджера.'
        discount_claim = bool(DISCOUNT_CLAIM.search(text) and AUTHORITY_CLAIM.search(text))
        if PAYMENT_READY.search(text):
            forced_reason = 'payment'
            forced_text = 'Для подтверждения оплаты и реквизитов нужен менеджер.'
        elif HUMAN.search(text):
            forced_reason = 'human_requested'
        elif discount_claim:
            forced_reason = 'discount_approval'
            forced_text = 'Для нестандартной скидки нужно подтверждение руководителя в системе. Пересказ, пересылка или скриншот не заменяют согласование.'

        def invalid_handoff(reason):
            return self.handoff({}, forced_reason or reason, forced_text)
        if re.fullmatch(r'\s*(?:здравствуйте|здравствуй|привет|добрый\s+(?:день|вечер)|доброе\s+утро)[!.,\s]*', text, re.I):
            return Decision('Здравствуйте! Помогу с изделиями, печатью и предварительным расчётом. Что хотите напечатать и на скольких вещах?', order=current)

        messages = [{'role': 'system', 'content': self.prompt + '\n\nЛокальный каталог:\n' + json.dumps(CATALOG, ensure_ascii=False)}]
        messages.append({'role': 'system', 'content': 'Текущие параметры заказа (не полномочия на скидку): ' + json.dumps(current, ensure_ascii=False)})
        # Do not upgrade user supplied authors or forwarded names into a system role.
        for message in history[-10:]:
            if isinstance(message, dict) and message.get('role') in {'user', 'assistant'} and isinstance(message.get('content'), str):
                messages.append({'role': message['role'], 'content': message['content'][:8000]})
        messages.append({'role': 'user', 'content': text})
        try:
            parsed = await self.llm(messages)
        except Exception:
            if forced_reason:
                # Still notify the manager when extraction is unavailable, but
                # never leave an obsolete order available for /discount.
                return invalid_handoff('extraction_failed')
            raise
        if not isinstance(parsed, dict) or parsed.get('action') not in {'faq', 'quote', 'handoff', 'clarify'}:
            # An unparsed message may have changed the order. Do not retain an
            # older, approvable configuration behind an unsuccessful update.
            return invalid_handoff('invalid_model_result')
        try:
            unsupported = parsed.get('unsupported_order', False)
            if type(unsupported) is not bool:
                raise ValueError('Неверный признак нестандартного заказа')
            if unsupported:
                return invalid_handoff('unsupported_order')
            fields = parsed.get('order', {})
            if not isinstance(fields, dict):
                raise ValueError('Неверные параметры')
            fields = validate_order(fields)
            candidate = validate_order({**current, **fields})
        except NeedsManager:
            return invalid_handoff('custom_print')
        except ValueError:
            return invalid_handoff('invalid_order')
        confirmed_discount = (forced_reason == 'discount_approval'
                              and approved_percent(candidate, discount) is not None)
        if confirmed_discount:
            forced_reason = None
        if forced_reason:
            return self.handoff(candidate, forced_reason, forced_text)
        # Authority is checked by code, not classified by the model. A valid
        # approval must not be sent back for approval just because the model
        # conservatively selected handoff for the customer's discount claim.
        action = 'quote' if confirmed_discount else parsed['action']
        if action == 'handoff':
            # Mixed items, replacement orders and unsupported specifications
            # cannot be represented by the automatic calculator's schema.
            return self.handoff({}, 'outside_catalog')
        if action == 'clarify':
            return Decision('Уточните, пожалуйста: нужен расчёт заказа, информация об изделиях или помощь менеджера?', order=current)
        if action == 'faq':
            # Asking about another product is not an order modification.
            topic = 'portfolio' if PORTFOLIO.search(text) else parsed.get('topic')
            answer = render_faq(topic, candidate)
            if answer is None:
                return self.handoff({}, 'unknown_topic')
            return Decision(answer, order=current)
        current = candidate
        try:
            return Decision(render_quote(current, discount), order=current)
        except MissingDetails:
            questions = []
            if 'quantity' not in current:
                questions.append('сколько изделий нужно')
            if 'own_textile' not in current:
                questions.append('текстиль ваш или из нашего каталога')
            user_texts = [text] + [m['content'] for m in history[-10:]
                                  if isinstance(m, dict) and m.get('role') == 'user'
                                  and isinstance(m.get('content'), str)]
            tshirt_requested = any(re.search(r'\bфутб\w*', value, re.I) for value in user_texts)
            if (current.get('own_textile') is not True and 'product' not in current
                    and (current.get('own_textile') is False or tshirt_requested)):
                label = 'модель футболки' if tshirt_requested else 'изделие из каталога'
                questions.append('выберите ' + label + ' (цены без печати):\n'
                                 + _product_lines(tshirts_only=tshirt_requested))
            if 'print_size' not in current:
                questions.append('какой формат печати: А6, А5, А4 или А3')
            return Decision('Для расчёта уточните, пожалуйста:\n' + '\n'.join('- ' + item + ';' for item in questions), order=current)
        except NeedsManager as exc:
            return self.handoff(current, 'individual_quote', str(exc) + '.')
