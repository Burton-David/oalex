"""oalex — async Python client for the OpenAlex scholarly works API.

Quick start::

    import asyncio
    from oalex import Client

    async def main() -> None:
        async with Client(email="you@example.com") as oa:
            works = await oa.search("attention is all you need", per_page=5)
            for w in works:
                print(w.id, w.title)

    asyncio.run(main())

See https://docs.openalex.org/ for the underlying API reference.
"""

from oalex.client import Client
from oalex.errors import OalexError, OalexUnavailable
from oalex.types import Author, Work, parse_work

__all__ = [
    "Author",
    "Client",
    "OalexError",
    "OalexUnavailable",
    "Work",
    "parse_work",
]

__version__ = "0.1.0"
