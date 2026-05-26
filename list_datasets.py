from huggingface_hub import list_datasets

# List first 20 datasets
datasets = list_datasets(limit=20)
for d in datasets:
    print(d.id)
