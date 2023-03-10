from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class PersonaGPT:
    DEFAULT_PERSONAS = [
        "I like to play guitar.",
        "I hate onions.",
        "I am 20 years old.",
    ]

    def __init__(self, personas=DEFAULT_PERSONAS):
        self.tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT", padding_side='left')
        self.model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.personas = [fact + self.tokenizer.eos_token for fact in personas]
        self.personas = self.tokenizer.encode(
            ''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))

        self.dialog_history = []

    def clear_history(self):
        self.dialog_history = []

    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def to_data(self, x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()

    def to_var(self, x):
        if not torch.is_tensor(x):
            x = torch.Tensor(x)
        if torch.cuda.is_available():
            x = x.cuda()
        return x

    def display_dialog_history(self, dialog_hx):
        for j, line in enumerate(dialog_hx):
            msg = self.tokenizer.decode(line)
            if j % 2 == 0:
                print(">> User: " + msg)
            else:
                print("Bot: " + msg)
                print()

    def generate_next(self, bot_input_ids, do_sample=True, top_k=10, top_p=.92,
                      max_length=1000):


        full_msg = self.model.generate(bot_input_ids, do_sample=do_sample,
                                                  top_k=top_k, top_p=top_p,
                                                  max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        msg = self.to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
        return msg

    def __call__(self, user_input):
        # encode the user input
        user_input = self.tokenizer.encode(user_input + self.tokenizer.eos_token)
        # append to the chat history
        self.dialog_history.append(user_input)

        # generated a response while limiting the total chat history to 1000 tokens,
        bot_input_ids = self.to_var([self.personas + self.flatten(self.dialog_history)]).long()
        msg = self.generate_next(bot_input_ids)
        self.dialog_history.append(msg)

        return self.tokenizer.decode(msg, skip_special_tokens=True)


if __name__ == "__main__":
    chatbot_model = PersonaGPT()
    for step in range(8):
        user_input = input(">> User: ")
        print("<< Bot:", chatbot_model(user_input))
