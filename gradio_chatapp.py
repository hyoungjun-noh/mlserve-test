import gradio as gr
import random
import personaGPT
from ray.serve.gradio_integrations import GradioServer
from ray import serve

def gradio_builder():
    chatbot_model = personaGPT.PersonaGPT()

    def respond(chat_history, message):
        response = chatbot_model(message)
        # response = random.choice(["Yes", "No"])
        return chat_history + [[message, response]]

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear Chat")

        msg.submit(respond, [chatbot, msg], chatbot)
        clear.click(lambda: chatbot_model.clear_history(), None, chatbot, queue=False)

    return demo

# demo = gradio_builder()
# demo.launch()

app = GradioServer.options(num_replicas=1, ray_actor_options={"num_cpus": 1}).bind(gradio_builder)
# serve.run(app)
