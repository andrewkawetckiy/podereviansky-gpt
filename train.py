import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import safetensors.torch  # Для збереження у форматі safetensors
import json  # Для збереження конфігурації

# Завантаження текстів
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.vocab = sorted(set(text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        self.encoded = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return len(self.encoded) - self.seq_length

    def __getitem__(self, idx):
        x = self.encoded[idx:idx + self.seq_length]
        y = self.encoded[idx + 1:idx + self.seq_length + 1]
        return x, y

# Модель RNN
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
        return (torch.zeros(2, batch_size, self.hidden_dim).to(device),
                torch.zeros(2, batch_size, self.hidden_dim).to(device))

# Параметри
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

text_dir = "D:/UNIVERSITY/DEEPLEARN/m1v7/m1v7/texts"
seq_length = 100
batch_size = 4096  # Оптимізовано для 16 ГБ GPU (128)
hidden_dim = 256
num_layers = 2
iterations = 2500
save_interval = 500  # Збереження чекпойнтів кожні 500 ітерацій
clip_value = 1.0  # Gradient Clipping

# Функція для збереження моделі
def save_model_safetensors(model, dataset, config, step):
    safetensors.torch.save_file(model.state_dict(), f"model_checkpoint_step_{step}.safetensors")
    torch.save(dataset.char_to_idx, f"model_checkpoint_step_{step}_char_to_idx.pth")
    torch.save(dataset.idx_to_char, f"model_checkpoint_step_{step}_idx_to_char.pth")
    with open(f"model_checkpoint_step_{step}_config.json", "w") as f:
        json.dump(config, f)
    print(f"Checkpoint saved at step {step}.")

try:
    # Завантаження текстів
    print("Loading texts...")
    all_text = ""
    for i in range(1, 21):
        with open(os.path.join(text_dir, f"play{i}.txt"), "r", encoding="utf-8") as f:
            all_text += f.read()
    print(f"Loaded {len(all_text)} characters.")

    dataset = TextDataset(all_text, seq_length)
    print(f"Vocabulary size: {len(dataset.vocab)}")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # Модель, лосс та оптимізатор
    model = RNNModel(len(dataset.vocab), hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    # Конфігурація моделі
    config = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "vocab_size": len(dataset.vocab),
        "seq_length": seq_length
    }

    # Тренування
    print("Starting training...")
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(range(1, iterations + 1), desc="Training")
    for step in progress_bar:
        hidden = model.init_hidden(batch_size)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            inputs = nn.functional.one_hot(inputs, num_classes=len(dataset.vocab)).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs.view(-1, len(dataset.vocab)), targets.view(-1))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()

            hidden = tuple(h.detach() for h in hidden)
            epoch_loss += loss.item()

        if step % 10 == 0:
            avg_loss = epoch_loss / 10
            progress_bar.set_postfix(loss=avg_loss)
            epoch_loss = 0

        if step % save_interval == 0:
            save_model_safetensors(model, dataset, config, step)

    print("Training complete.")
    save_model_safetensors(model, dataset, config, iterations)

except FileNotFoundError as fnf_error:
    print(f"File not found: {fnf_error}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
