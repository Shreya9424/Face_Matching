import face_recognition
import pickle
import os

# 1. Setup the paths
folder_path = 'photos'
save_file = 'my_database.pkl'

known_encodings = []
names = []

print("Step 1: Reading all 100 images... Please wait.")

# 2. Loop through every file in the 'photos' folder
files = os.listdir(folder_path)

for file_name in files:
    # Only look at image files
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        
        # Create the full path (e.g., photos/john.jpg)
        image_path = os.path.join(folder_path, file_name)
        
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Detect the face and turn it into math (encoding)
        # We assume there is 1 face per image
        face_math = face_recognition.face_encodings(image)
        
        # If a face was found, save it
        if len(face_math) > 0:
            known_encodings.append(face_math[0])
            # Save the name (remove .jpg from the name)
            clean_name = os.path.splitext(file_name)[0]
            names.append(clean_name)
            print(f"Processed: {clean_name}")

# 3. Save everything to a file
data = {"encodings": known_encodings, "names": names}

f = open(save_file, "wb")
f.write(pickle.dumps(data))
f.close()

print(f"Done! Saved {len(names)} people to {save_file}")