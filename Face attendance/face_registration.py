import cv2
import dlib
import time
import csv
import numpy as np

class FaceRegistration:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.shape_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.face_descriptor_extractor = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    def capture_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (self.width - 50, self.height - 50))
        frame = cv2.flip(frame, 1)
        return frame

    def register_faces(self, id, name, count, interval):
        start_time = time.time()
        collect_count = 0
        f = open('C:\\Users\\Z16\\Desktop\\CV2\\data\\face.csv', 'a', newline="")
        csv_writer = csv.writer(f)

        while True:
            frame = self.capture_frame()

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = self.hog_face_detector(frame, 1)

            for face in detections:
                l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
                points = self.shape_detector(frame, face)

                for point in points.parts():
                    cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

                if collect_count < count:
                    now_time = time.time()

                    if now_time - start_time > interval:
                        face_descriptor = self.face_descriptor_extractor.compute_face_descriptor(frame, points)
                        face_descriptor = [i for i in face_descriptor]

                        line = [id, name, face_descriptor]
                        csv_writer.writerow(line)

                        collect_count += 1
                        start_time = now_time
                        print('collect count: {collect_count}'.format(collect_count=collect_count))
                    else:
                        pass
                else:
                    print('collect finished!')
                    return

            cv2.imshow("Face window", frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break

        f.close()
        self.cap.release()
        cv2.destroyAllWindows()

def get_feature_list():
    id_list = []
    name_list = []
    feature_list = None

    with open('C:\\Users\\Z16\\Desktop\\CV2\\data\\face.csv', 'r') as f:
        csv_reader = csv.reader(f)

        for line in csv_reader:
            id = line[0]
            name = line[1]
            id_list.append(id)
            name_list.append(name)
            # string change to list
            face_descriptor = eval(line[2])
            face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
            face_descriptor = np.reshape(face_descriptor, (1, -1))

            if feature_list is None:
                feature_list = face_descriptor
            else:
                feature_list = np.concatenate((feature_list, face_descriptor), axis=0)

    return id_list, name_list, feature_list
