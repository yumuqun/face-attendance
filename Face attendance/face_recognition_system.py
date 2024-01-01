from face_registration import FaceRegistration
from face_recognition import FaceRecognizer

def run_face_registration(id, name, count, interval):
    face_registrator = FaceRegistration()
    face_registrator.register_faces(id, name, count, interval)

def run_face_recognition(threshold=0.65):
    face_recognizer = FaceRecognizer(threshold=threshold)
   
