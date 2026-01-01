import face_recognition
import faiss
import pickle
import pandas as pd
import numpy as np
import os

# Files
IMAGE_FOLDER = 'photos'
CSV_FILE = 'people_details.csv'
INDEX_FILE = 'face_index.bin'
METADATA_FILE = 'metadata.pkl'

# 1. Load the details CSV
if not os.path.exists(CSV_FILE):
    print("Error: CSV file missing. Run create_fake_data.py first.")
    exit()

df = pd.read_csv(CSV_FILE)

known_face_encodings = []
metadata_list = []

print("Processing images... (This prepares the system for 1 Lakh+ scale)")

# 2. Process every row in the CSV
for index, row in df.iterrows():
    filename = row['image_filename']
    img_path = os.path.join(IMAGE_FOLDER, filename)
    
    if os.path.exists(img_path):
        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                # Add encoding to list
                known_face_encodings.append(encodings[0])
                
                # Add details to list (Store everything in the row)
                metadata_list.append(row.to_dict())
                print(f"Encoded: {filename}")
        except Exception as e:
            print(f"Skipping {filename}: {e}")

# 3. Create FAISS Index (The Speed Secret)
# Convert list to numpy array
encoding_matrix = np.array(known_face_encodings).astype('float32')

# Create the index (L2 distance is standard for faces)
dimension = 128
index = faiss.IndexFlatL2(dimension)
index.add(encoding_matrix)

# 4. Save Everything
faiss.write_index(index, INDEX_FILE)
with open(METADATA_FILE, 'wb') as f:
    pickle.dump(metadata_list, f)

print(f"\nDONE! Database built with {index.ntotal} people.")
print("You are ready for 1 Lakh images now.")