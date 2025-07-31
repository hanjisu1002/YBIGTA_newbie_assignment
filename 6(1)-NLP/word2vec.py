import torch
from torch import nn, Tensor
from torch.optim import Adam
from transformers import PreTrainedTokenizer
from typing import Literal

# 구현하세요!
def is_valid_context(context: list[int], pad_token_id: int) -> bool:
    return pad_token_id not in context

class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        self.vocab_size = vocab_size
        self.d_model = d_model

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(self, corpus: list[str], tokenizer: PreTrainedTokenizer, lr: float, num_epochs: int):
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        pad_token_id = tokenizer.pad_token_id

        tokenized_corpus = [
            tokenizer.encode(sentence, add_special_tokens=False)
            for sentence in corpus
        ]

        for epoch in range(num_epochs):
            total_loss = 0.0

            for tokens in tokenized_corpus:
                if len(tokens) < 2:
                    continue

                if self.method == "cbow":
                    loss = self._train_cbow(tokens, criterion, pad_token_id)
                elif self.method == "skipgram":
                    loss = self._train_skipgram(tokens, criterion, pad_token_id)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    def _train_cbow(self, tokens: list[int], criterion, pad_token_id: int):
        window = self.window_size
        losses = []
        for center_idx in range(window, len(tokens) - window):
            context = tokens[center_idx - window:center_idx] + tokens[center_idx + 1:center_idx + window + 1]
            target = tokens[center_idx]

            if not is_valid_context(context, pad_token_id) or target == pad_token_id:
                continue

            context_tensor = torch.LongTensor(context)  # (2 * window,)
            context_vec = self.embeddings(context_tensor)  # (2 * window, d_model)
            context_mean = context_vec.mean(dim=0, keepdim=True)  # (1, d_model)

            logits = self.weight(context_mean)  # (1, vocab_size)
            loss = criterion(logits.squeeze(0), torch.LongTensor([target]))  # target: scalar

            losses.append(loss)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True)

    def _train_skipgram(self, tokens: list[int], criterion, pad_token_id: int):
        window = self.window_size
        losses = []
        for center_idx in range(len(tokens)):
            center = tokens[center_idx]

            if center == pad_token_id:
                continue

            for offset in range(-window, window + 1):
                if offset == 0 or not (0 <= center_idx + offset < len(tokens)):
                    continue
                context = tokens[center_idx + offset]
                if context == pad_token_id:
                    continue

                center_tensor = torch.LongTensor([center])  # (1,)
                center_vec = self.embeddings(center_tensor)  # (1, d_model)
                logits = self.weight(center_vec).squeeze(0)  # shape: (vocab_size,)
                loss = criterion(logits.unsqueeze(0), torch.LongTensor([context]))  # shape: (1, vocab_size), (1,)

                losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True)
    
