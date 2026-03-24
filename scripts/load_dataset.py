from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

print("Loading dataset ...")

dataset = load_dataset("ashraq/fashion-product-images-small")

print("Dataset loaded successfully!")
print(dataset)

print("\nFirst product example:")
print(dataset["train"][0])

products = dataset["train"]


output_dir = Path("docs/products")
image_dir = Path("docs/images")

output_dir.mkdir(parents=True, exist_ok=True)
image_dir.mkdir(parents=True, exist_ok=True)

print(f"\nCreating {len(products)} product markdown files and images ...")

for i, product in tqdm(enumerate(products), total=len(products)):
    image_filename = f"product_{i+1}.jpg"
    image_path = image_dir / image_filename

    # save image
    product["image"].save(image_path)

    # create markdown content
    content = f"""Product Name: {product['productDisplayName']}
Gender: {product['gender']}
Category: {product['masterCategory']}
Sub Category: {product['subCategory']}
Article Type: {product['articleType']}
Color: {product['baseColour']}
Season: {product['season']}
Year: {product['year']}
Usage: {product['usage']}
Image Path: {image_path.as_posix()}
"""

    # save markdown
    file_path = output_dir / f"product_{i+1}.md"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("\nAll product documents created successfully!")
print(f"Markdown files saved in: {output_dir}")
print(f"Images saved in: {image_dir}")