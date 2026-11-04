"""Validated order data and deterministic money calculations; no model prices."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import time

CATALOG = json.loads(Path(__file__).with_name('catalog.json').read_text(encoding='utf-8'))
FIELDS = frozenset({'product', 'own_textile', 'quantity', 'print_size', 'placements', 'color', 'size'})
CENT = Decimal('0.01')


class MissingDetails(ValueError):
    """Required quote fields are not yet known."""


class NeedsManager(ValueError):
    """The catalog does not define a safe automatic calculation."""


def validate_order(order: dict | None) -> dict:
    """Normalize partial orders. Reject unknown fields and invalid scalar values."""
    if order is None:
        return {}
    if not isinstance(order, dict) or set(order) - FIELDS:
        raise ValueError('Неизвестные параметры заказа')
    result = {}
    for key, value in order.items():
        if value is None:
            continue
        if key == 'product':
            if not isinstance(value, str) or value not in CATALOG['products']:
                raise ValueError('Неизвестное изделие')
        elif key == 'own_textile':
            if type(value) is not bool:
                raise ValueError('Нужно уточнить, чей текстиль')
        elif key in {'quantity', 'placements'}:
            if type(value) is not int or not 1 <= value <= 100000:
                raise ValueError('Количество должно быть целым положительным числом')
        elif key == 'print_size':
            if not isinstance(value, str):
                raise ValueError('Неверный формат печати')
            value = value.upper().replace('А', 'A').strip()
            if value not in CATALOG['print_prices']:
                raise NeedsManager('Размер печати отсутствует в прайсе')
        elif key in {'color', 'size'}:
            if not isinstance(value, str) or not value.strip() or len(value) > 80:
                raise ValueError('Неверный цвет или размер изделия')
            value = value.strip().lower() if key == 'color' else value.strip().upper()
        result[key] = value
    return result


def order_fingerprint(order: dict) -> str:
    normalized = validate_order(order)
    # One print is the explicit standard assumption displayed in every quote.
    normalized.setdefault('placements', 1)
    encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def approved_percent(order: dict, discount: dict | None, now: float | None = None) -> Decimal | None:
    if not isinstance(discount, dict):
        return None
    try:
        percent = Decimal(str(discount['percent']))
        expires_at = float(discount['expires_at'])
        if not percent.is_finite() or not Decimal(0) <= percent <= Decimal(100):
            return None
        if not expires_at > (time.time() if now is None else now):
            return None
        if discount['order_fingerprint'] != order_fingerprint(order):
            return None
        return percent
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return None


@dataclass(frozen=True)
class Quote:
    order: dict
    quantity: int
    discount_percent: Decimal
    approved_discount: bool
    textile_before: Decimal
    textile_after: Decimal
    printing_before: Decimal
    printing_after: Decimal
    minimum_adjustment: Decimal
    total: Decimal


def calculate_quote(order: dict, discount: dict | None = None, now: float | None = None) -> Quote:
    order = validate_order(order)
    required = ['quantity', 'own_textile', 'print_size']
    if order.get('own_textile') is False:
        required.append('product')
    missing = [key for key in required if key not in order]
    if missing:
        raise MissingDetails(', '.join(missing))
    if order.get('placements', 1) != 1:
        raise NeedsManager('Стоимость нескольких нанесений должен подтвердить менеджер')
    product = CATALOG['products'].get(order.get('product'), {})
    if order.get('product') in {'cap', 'visor'}:
        raise NeedsManager('Для головного убора нужно согласовать размер и цену нанесения')
    own = order['own_textile']
    if not own and product.get('price') is None:
        raise NeedsManager('Для этого изделия нет фиксированной цены')
    quantity = order['quantity']
    percent = approved_percent(order, discount, now)
    trusted = percent is not None
    if percent is None:
        if quantity > 20:
            raise NeedsManager('Для тиража более 20 изделий нужен индивидуальный расчёт')
        percent = next(Decimal(tier['percent']) for tier in CATALOG['standard_discounts']
                       if tier['min'] <= quantity <= tier['max'])
    textile_before = (Decimal(0) if own else Decimal(product['price'])) * quantity
    printing_before = Decimal(CATALOG['print_prices'][order['print_size']]) * quantity
    factor = (Decimal(100) - percent) / Decimal(100)
    # Round each component total once, in kopecks; never use binary floats for money.
    textile_after = (textile_before * factor).quantize(CENT, rounding=ROUND_HALF_UP)
    printing_after = (printing_before * factor).quantize(CENT, rounding=ROUND_HALF_UP)
    subtotal = textile_after + printing_after
    minimum_adjustment = max(Decimal(0), Decimal(CATALOG['own_textile_minimum']) - subtotal) if own else Decimal(0)
    return Quote(order, quantity, percent, trusted, textile_before, textile_after,
                 printing_before, printing_after, minimum_adjustment, subtotal + minimum_adjustment)


def money(value: Decimal) -> str:
    text = f'{value:,.2f}'.replace(',', ' ').replace('.', ',')
    return text[:-3] if text.endswith(',00') else text
