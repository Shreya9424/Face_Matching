import streamlit as st
import face_recognition
import faiss
import pickle
import numpy as np
from PIL import Image
import os

# --- LOAD RESOURCES ---
@st.cache_resource
def load_system():
    try:
        index = faiss.read_index('face_index.bin')
        with open('metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata
    except:
        return None, None

index, metadata = load_system()

# --- APP LAYOUT ---
st.set_page_config(layout="wide", page_title="Face Match System")
st.title("🔒 1 Lakh+ Scale Facial Recognition")

if index is None:
    st.error("System files not found. Please run create_smart_database.py")
else:
    col_upload, col_result = st.columns([1, 2])

    with col_upload:
        st.subheader("1. Upload Photo")
        uploaded_file = st.file_uploader("Upload passport photo", type=['jpg', 'png', 'jpeg'])

        if uploaded_file:
            img_uploaded = Image.open(uploaded_file)
            st.image(img_uploaded, caption="Uploaded Image", width=250)
            img_array = np.array(img_uploaded)

    with col_result:
        if uploaded_file:
            st.subheader("2. Search Results")
            with st.spinner("Scanning Database..."):
                
                # Detect Face
                face_locations = face_recognition.face_locations(img_array, model='hog')
                face_encodings = face_recognition.face_encodings(img_array, face_locations)

                if len(face_encodings) > 0:
                    target_encoding = face_encodings[0].astype('float32').reshape(1, -1)
                    
                    # Search
                    D, I = index.search(target_encoding, 1)
                    best_match_index = I[0][0]
                    distance = D[0][0]
                    
                    # --- UPDATED STRICT THRESHOLD ---
                    # Old value: 0.45 (Too friendly)
                    # New value: 0.35 (Very Strict)
                    # Result: Matches like 0.39 will now be rejected.
                    STRICT_THRESHOLD = 0.35

                    if distance < STRICT_THRESHOLD:
                        person_info = metadata[best_match_index]
                        
                        st.success("✅ MATCH FOUND!")
                        st.write(f"Match Score: {round(distance, 3)} (Excellent Match)")
                        
                        result_cols = st.columns(2)
                        with result_cols[0]:
                            st.write("**Database Photo:**")
                            db_image_path = os.path.join('photos', person_info['image_filename'])
                            if os.path.exists(db_image_path):
                                st.image(db_image_path, width=200)
                            else:
                                st.write("Image file missing")

                        with result_cols[1]:
                            # --- DETAILS ---
                            st.write("### Personal Details:")
                            st.write(f"👤 **Name:** {person_info['Name']}")
                            st.write(f"💼 **Profession:** {person_info['Profession']}")
                            st.write(f"🎂 **Age:** {person_info['Age']}")
                            st.write(f"🩸 **Blood Group:** {person_info['Blood_Group']}")
                            st.write(f"📏 **Height:** {person_info['Height']}")
                            st.write(f"⚖️ **Weight:** {person_info['Weight']}")

                        # --- DOWNLOAD TEXT FILE ---
                        report_text = f"""
OFFICIAL IDENTIFICATION REPORT
------------------------------
STATUS: MATCH FOUND
------------------------------
Name: {person_info['Name']}
Profession: {person_info['Profession']}
Age: {person_info['Age']}
Blood Group: {person_info['Blood_Group']}
Height: {person_info['Height']}
Weight: {person_info['Weight']}
------------------------------
Match Confidence Score: {distance}
"""
                        st.download_button(
                            label="📥 Download Full Report (.txt)",
                            data=report_text,
                            file_name=f"{person_info['Name']}_Report.txt",
                            mime="text/plain"
                        )
                    
                    else:
                        st.error("❌ NO MATCH FOUND")
                        st.warning("This person is not in our database.")
                        st.write(f"Closest match score was: {round(distance, 3)}")
                        st.caption(f"(We rejected it because it was higher than {STRICT_THRESHOLD})")
                        
                else:
                    st.warning("⚠️ No face detected in uploaded photo.")