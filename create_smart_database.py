import face_recognition
import faiss
import pickle
import numpy as np
import os
import random

# --- CONFIGURATION ---
REAL_IMAGE_FOLDER = 'photos'
TOTAL_SIZE = 100000
INDEX_FILE = 'face_index.bin'
METADATA_FILE = 'metadata.pkl'

all_vectors = []
metadata_list = []
image_files = [f for f in os.listdir(REAL_IMAGE_FOLDER) if f.endswith(('.jpg', '.png'))]

print("STEP 1: Processing REAL images...")

# Lists for random details
professions = ["Software Engineer", "Doctor", "Civil Engineer", "Manager", "Teacher", "Pilot", "Data Scientist", "Lawyer"]
blood_groups = ["A+", "B+", "O+", "AB+", "O-", "A-"]

# --- PART A: REAL IMAGES ---
count_real = 0
for filename in image_files:
    path = os.path.join(REAL_IMAGE_FOLDER, filename)
    try:
        img = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(img)
        
        if len(encodings) > 0:
            all_vectors.append(encodings[0])
            name = os.path.splitext(filename)[0]
            
            # ADDING ALL DETAILS HERE
            metadata_list.append({
                "image_filename": filename,
                "Name": name,
                "Height": f"{random.randint(160, 190)} cm",
                "Weight": f"{random.randint(60, 90)} kg",
                "Age": random.randint(22, 60),               # Added Age
                "Blood_Group": random.choice(blood_groups),  # Added Blood Group
                "Profession": random.choice(professions)     # Added Profession
            })
            count_real += 1
            print(f"Encoded: {filename}")
    except:
        pass

# --- PART B: FAKE DATA ---
remaining = TOTAL_SIZE - count_real
print(f"STEP 2: Generating {remaining} fake entries...")

if remaining > 0:
    fake_vectors = np.random.rand(remaining, 128).astype('float32')
    real_vectors_np = np.array(all_vectors).astype('float32')
    combined_vectors = np.vstack((real_vectors_np, fake_vectors))
    
    for i in range(remaining):
        random_image = random.choice(image_files)
        metadata_list.append({
            "image_filename": random_image,
            "Name": f"Person_{i}",
            "Height": "170 cm",
            "Weight": "70 kg",
            "Age": random.randint(20, 50),
            "Blood_Group": random.choice(blood_groups),
            "Profession": random.choice(professions)
        })
else:
    combined_vectors = np.array(all_vectors).astype('float32')

# --- SAVE ---
index = faiss.IndexFlatL2(128)
index.add(combined_vectors)
faiss.write_index(index, INDEX_FILE)
with open(METADATA_FILE, 'wb') as f:
    pickle.dump(metadata_list, f)

print("SUCCESS! Database updated with Professions, Age, and Blood Group.")