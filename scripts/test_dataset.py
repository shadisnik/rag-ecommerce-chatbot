print("STARTING...")

from datasets import load_dataset

print("BEFORE LOAD")

dataset = load_dataset("ashraq/fashion-product-images-small", split="train[:2]")

print("AFTER LOAD")

print("\nFIRST ITEM:")
print(dataset[0])

print("\nKEYS:")
print(dataset[0].keys())