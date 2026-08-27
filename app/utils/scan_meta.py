"""Scan id generation and Indonesian date/time formatting."""

import secrets
from datetime import date, datetime, timedelta

_ID_MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def new_id() -> str:
    return "A-" + secrets.token_hex(3).upper()


def day_group(dt: datetime) -> str:
    today = date.today()
    d = dt.date()
    if d == today:
        return "HARI INI"
    if d == today - timedelta(days=1):
        return "KEMARIN"
    return f"{d.day:02d} {_ID_MONTHS[d.month - 1]}"


def date_label(dt: datetime) -> str:
    return f"{dt.day:02d} {_ID_MONTHS[dt.month - 1]} · {dt.strftime('%H:%M')}"
