"""Arthiq AI service.

A standalone analysis service. It receives structured market and technical
context from the Arthiq backend, asks a configured AI provider for a reading,
and returns a validated, provider-neutral :class:`TradingDecision`.

It holds no broker credentials, opens no database connection, and cannot place
an order: nothing in this package imports a broker client, a database driver or
an authentication SDK. The backend remains the system of record and the only
authority for trading.
"""

__version__ = "0.1.0"
