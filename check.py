import face_recognition
known_image = face_recognition.load_image_file("uploads/f15922f7-a0fe-4eea-9cdf-874a7499e8b7.jpeg")
unknown_image = face_recognition.load_image_file("uploads/ed555e0e-2733-493d-9f36-7f411894ef8f.jpeg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

print(results)