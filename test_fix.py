try:
    import face_recognition_models
    print("SUCCESS: The models are found!")
except Exception as e:
    print("REAL ERROR:", e)