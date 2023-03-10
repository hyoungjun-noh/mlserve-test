import starlette.requests
import requests
from ray import serve


@serve.deployment
class Counter:
    def __call__(self, request: starlette.requests.Request):
        return request.query_params


counter = Counter.bind()