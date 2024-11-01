import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from torch.utils.data import Dataset, DataLoader

# Load a txt file and create a tag
def load_data(file_path, label):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Loaded {len(lines)} lines from {file_path}")  # 添加调试信息
        return [(line.strip(), label) for line in lines if line.strip()]
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []


# Load data on art and technology themes
# art_data = load_data("Art_creation_sentences.txt", label=1)  # label 1代表艺术
# tech_data = load_data("Experimentation_sentences.txt", label=0)  # label 0代表科技
art_data = load_data("/Users/zhouzhenzhou/Desktop/CS7980/Code/Art_creation_sentences.txt", label=1)
tech_data = load_data("/Users/zhouzhenzhou/Desktop/CS7980/Code/Experimentation_sentences.txt", label=0)


# Merge data and separate sentences and tags
data = art_data + tech_data
texts, labels = zip(*data)

# Existing art and technology sentence data
class ArtTechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Use tokenizer to encode the text
        inputs = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Make sure that the input dimension is correct and remove unnecessary dimensions.
        inputs = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }
        return inputs


# Prepare the data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
texts = ["Example art sentence.", "Example tech sentence."]
labels = [1, 0]  # 1 represents art, 0 represents technology
dataset = ArtTechDataset(texts, labels, tokenizer)

# Load a pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Fine tune settings
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Fine-tuning using Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Start fine-tuning
trainer.train()


# Specify Save Path
model_path = "./results"
# Load fine-tuned models and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)