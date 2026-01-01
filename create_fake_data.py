import os
import pandas as pd
import random

# Settings
FOLDER_PATH = 'photos'
CSV_FILE = 'people_details.csv'

# Get list of images
files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(('.jpg', '.png'))]

data = []
print("Generating fake details for your images...")

for filename in files:
    name = os.path.splitext(filename)[0] # Removes .jpg
    
    # Create fake details
    height = random.randint(150, 190)
    weight = random.randint(50, 90)
    age = random.randint(20, 60)
    blood_group = random.choice(['A+', 'B+', 'O+', 'AB+'])
    
    data.append({
        "image_filename": filename,
        "Name": name,
        "Height": f"{height} cm",
        "Weight": f"{weight} kg",
        "Age": age,
        "Blood_Group": blood_group
    })

# Save to CSV (Excel format)
df = pd.DataFrame(data)
df.to_csv(CSV_FILE, index=False)
print(f"Success! Created {CSV_FILE} with details for {len(files)} people.")