"""Script to create synthetic test data for model testing."""
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/raw/images").mkdir(parents=True, exist_ok=True)

print("Creating synthetic test data...")

# Create synthetic text data (X_raw.csv)
n_samples = 100
product_ids = range(1000, 1000 + n_samples)
image_ids = range(2000, 2000 + n_samples)

designations = [
    f"Product {i} - test designation with some keywords"
    for i in range(n_samples)
]

descriptions = [
    f"This is a detailed description for product {i}. It contains multiple words and sentences."
    for i in range(n_samples)
]

X_data = pd.DataFrame({
    'productid': product_ids,
    'imageid': image_ids,
    'designation': designations,
    'description': descriptions
})

X_data.to_csv("data/raw/X_raw.csv")
print(f"✓ Created X_raw.csv with {n_samples} samples")

# Create synthetic labels (Y_raw.csv)
# 27 classes as in Rakuten dataset
classes = [1000 + i*40 for i in range(27)]
y_data = pd.DataFrame({
    'prdtypecode': np.random.choice(classes, size=n_samples)
})

y_data.to_csv("data/raw/Y_raw.csv")
print(f"✓ Created Y_raw.csv with {len(classes)} classes")

# Create synthetic images
print("Creating synthetic images...")
for i, (product_id, image_id) in enumerate(zip(product_ids, image_ids)):
    # Create random image
    img_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save with Rakuten naming convention
    img_path = f"data/raw/images/image_{image_id}_product_{product_id}.jpg"
    img.save(img_path)
    
    if (i + 1) % 20 == 0:
        print(f"  Created {i + 1}/{n_samples} images")

print(f"✓ Created {n_samples} synthetic images")
print("\n" + "="*60)
print("Test data creation completed!")
print("="*60)
print(f"\nFiles created:")
print(f"  - data/raw/X_raw.csv ({n_samples} rows)")
print(f"  - data/raw/Y_raw.csv ({n_samples} rows)")
print(f"  - data/raw/images/ ({n_samples} images)")
print(f"\nYou can now test the preprocessing and training pipelines.")
