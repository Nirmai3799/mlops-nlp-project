from datasets import load_dataset

# Load the Newsroom dataset
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

# Check available splits
print(dataset)

# Access train split
train_data = dataset["train"]

# Inspect a sample
print(train_data[0])
