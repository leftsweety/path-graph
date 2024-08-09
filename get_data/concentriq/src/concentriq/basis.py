import base64
import json
from threading import Lock

from requests import Session


class ThreadSafeRequestsSession(Session):
    def __init__(self, email: str, password: str):
        super().__init__()
        auth_header = 'Basic {}'.format(
            base64.b64encode(bytes(f'{email}:{password}', 'utf-8')).decode('utf-8'))
        self.headers.update({'Authorization': auth_header})
        self._lock = Lock()

    def get(self, *args, **kwargs) -> dict:
        with self._lock:
            response = super().get(*args, **kwargs)
            response.raise_for_status()
            return json.loads(response.text)

    def post(self, *args, **kwargs) -> dict:
        with self._lock:
            response = super().post(*args, **kwargs)
            response.raise_for_status()
            return json.loads(response.text)

    def patch(self, *args, **kwargs) -> dict:
        with self._lock:
            response = super().patch(*args, **kwargs)
            response.raise_for_status()
            return json.loads(response.text)

    def put(self, *args, **kwargs) -> dict:
        with self._lock:
            response = super().patch(*args, **kwargs)
            response.raise_for_status()
            return json.loads(response.text)

    def delete(self, *args, **kwargs) -> dict:
        with self._lock:
            response = super().delete(*args, **kwargs)
            response.raise_for_status()
            return json.loads(response.text)
