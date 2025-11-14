import contextvars

request_namespace = contextvars.ContextVar("request_namespace", default="A")