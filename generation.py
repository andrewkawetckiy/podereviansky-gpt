import torch
import torch.nn as nn
import safetensors.torch
import json
import gradio as gr

# Параметри пристрою
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Клас моделі (той самий, що й під час тренування)
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_size=vocab_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(num_layers, batch_size, self.hidden_dim),
                torch.zeros(num_layers, batch_size, self.hidden_dim))

# Завантаження словників
char_to_idx = torch.load("./models/model_checkpoint_step_500_char_to_idx.pth")
idx_to_char = torch.load("./models/model_checkpoint_step_500_idx_to_char.pth")
vocab_size = len(char_to_idx)

# Завантаження конфігурації моделі
with open("./models/model_checkpoint_step_500_config.json", "r") as f:
    config = json.load(f)

hidden_dim = config["hidden_dim"]
num_layers = config["num_layers"]

# Ініціалізація моделі
model = RNNModel(vocab_size, hidden_dim, num_layers)
print("Loading model weights...")

# Завантаження ваг на CPU, а потім перенесення на потрібний пристрій
state_dict = safetensors.torch.load_file("./models/model_checkpoint_step_500.safetensors")
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# Функція генерації тексту
def generate_text(model, start_string, gen_length=0, temperature=0.0):
    print(f"Using temperature: {temperature}")  # Додано для перевірки
    model.eval()
    input_eval = torch.tensor([char_to_idx[c] for c in start_string], dtype=torch.long).unsqueeze(0).to(device)
    input_eval = nn.functional.one_hot(input_eval, num_classes=vocab_size).float()
    hidden = (torch.zeros(num_layers, 1, hidden_dim).to(device),
              torch.zeros(num_layers, 1, hidden_dim).to(device))

    generated_text = start_string

    for _ in range(gen_length):
        with torch.no_grad():
            output, hidden = model(input_eval, hidden)
        output_dist = output[:, -1, :] / temperature  # Використання температури
        probabilities = torch.nn.functional.softmax(output_dist, dim=-1).squeeze()
        predicted_id = torch.multinomial(probabilities, num_samples=1).item()

        generated_text += idx_to_char[predicted_id]
        input_eval = torch.tensor([[predicted_id]], dtype=torch.long).to(device)
        input_eval = nn.functional.one_hot(input_eval, num_classes=vocab_size).float()

    return generated_text

# Функція для Gradio
def generate_text_interface(start_string, gen_length, temperature):
    print(f"Start string: {start_string}, Length: {gen_length}, Temperature: {temperature}")
    return generate_text(model, start_string, gen_length, temperature)


# Веб-інтерфейс
interface = gr.Interface(
    fn=generate_text_interface,
    inputs=[
        gr.Textbox(label="Початковий текст", placeholder="Введіть текст для початку генерації"),
        gr.Slider(50, 1000, value=150, step=10, label="Довжина генерації"),
        gr.Slider(0.1, 1.0, value=0.77, step=0.01, label="Температура")
    ],
    outputs=gr.Textbox(label="Згенерований текст"),
    title="Podervyansky-GPT",
    description="Введіть текст, оберіть параметри й отримайте згенерований текст."
)

# Запуск
if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860, share=False)

