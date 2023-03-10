import ray
import requests
from fastapi import FastAPI
from ray import serve
import gradio as gr
from gradio_chatapp import gradio_builder

fast_app = FastAPI()


@serve.deployment()
@serve.ingress(fast_app)
class App:
    @fast_app.get("/hello")
    def hello(self):
        return "Hello, world!"


gr.mount_gradio_app(fast_app, gradio_builder(), path="/gradio")


app = App.bind()


