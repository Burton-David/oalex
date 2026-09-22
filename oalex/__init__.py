"""oalex: async Python client for the OpenAlex scholarly works API.

Quick start::

    import asyncio
    from oalex import Client

    async def main() -> None:
        async with Client(api_key="...") as oa:
            works = await oa.search("attention is all you need", per_page=5)
            for w in works:
                print(w.id, w.title)

    asyncio.run(main())

See https://help.openalex.org/api/ for the underlying API reference.
"""

from oalex.client import Client
from oalex.errors import OalexError, OalexRateLimited, OalexRequestError, OalexUnavailable
from oalex.types import Author, Work, parse_work

__all__ = [
    "Author",
    "Client",
    "OalexError",
    "OalexRateLimited",
    "OalexRequestError",
    "OalexUnavailable",
    "Work",
    "parse_work",
]

__version__ = "0.2.0"
