# 구현하세요!
from datasets import load_dataset
def load_corpus() -> list[str]:
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")
    corpus = [example["verse_text"] for example in dataset if example["verse_text"].strip()]
    return corpus