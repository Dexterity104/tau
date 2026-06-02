from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx


class GitHubClient(ABC):
    @abstractmethod
    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def get(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def put(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def patch(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def delete(self, url: str, **kwargs: Any) -> httpx.Response: ...


