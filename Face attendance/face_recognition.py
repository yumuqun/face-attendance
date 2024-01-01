import cv2
import numpy as np
import dlib
import time
import csv
from face_registration import get_feature_list

class FaceRecognizer:
    def __init__(self, threshold=0.65):
        self.cap = cv2.VideoCapture(1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.shape_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.face_descriptor_extractor = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        self.threshold = threshold

        self.id_list, self.name_list, self.feature_list = get_feature_list()

        self.recog_record = {}
        self.f = open('C:\\Users\\Z16\\Desktop\\CV2\\data\\recognize.csv', 'a', newline="")
        self.csv_writer = csv.writer(self.f)

    def recognize_faces(self):
        while True:
            ret, frame = self.cap.read()

            frame = cv2.resize(frame, (self.width - 50, self.height - 50))
            frame = cv2.flip(frame, 1)

            detections = self.hog_face_detector(frame, 1)

            for face in detections:
                l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
                points = self.shape_detector(frame, face)

                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

                face_descriptor = self.face_descriptor_extractor.compute_face_descriptor(frame, points)
                face_descriptor = [f for f in face_descriptor]

                face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
                distances = np.linalg.norm((face_descriptor - self.feature_list), axis=1)
                min_index = np.argmin(distances)
                min_distance = distances[min_index]

                if min_distance < self.threshold:
                    predict_id = self.id_list[min_index]
                    predict_name = self.name_list[min_index]
                    now = time.time()
                    need_save = False
                    if predict_name in self.recog_record:
                        if now - self.recog_record[predict_name] > 3:
                            need_save = True
                            self.recog_record[predict_name] = now
                        else:
                            pass
                            need_save = False
                    else:
                        need_save = True
                        self.recog_record[predict_name] = now

                    if need_save:
                        time_local = time.localtime(self.recog_record[predict_name])
                        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
                        line = [predict_id, predict_name, min_distance, time_str]
                        self.csv_writer.writerow(line)
                        print("{time}: write successfully:{name}".format(name=predict_name, time=time_str))

            cv2.imshow("Face window", frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break

        self.f.close()
        self.cap.release()
        cv2.destroyAllWindows()
