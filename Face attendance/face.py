import cv2
import numpy as np
import dlib
import time
import csv




def faceRegister(id,name,count,interval):
    cap = cv2.VideoCapture(1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hog_face_detector = dlib.get_frontal_face_detector()
    shape_detector = dlib.shape_predictor('C:\\Users\\Z16\\Desktop\\CV2\\Face attendance\\shape_predictor_68_face_landmarks.dat')
    face_descriptor_extractor = dlib.face_recognition_model_v1('C:\\Users\\Z16\\Desktop\\CV2\\Face attendance\\dlib_face_recognition_resnet_model_v1.dat')
    
    strat_time = time.time()
    collect_count = 0
    f = open('C:\\Users\\Z16\\Desktop\\CV2\\Face attendance\\data\\face.csv','a',newline="")
    csv_writer = csv.writer(f)
    while True:
        ret,frame = cap.read()

        frame = cv2.resize(frame,(width-50,height-50))

        frame = cv2.flip(frame,1)

        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        detections = hog_face_detector(frame,1)

        for face in detections:
            l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
            points = shape_detector(frame,face)

            for point in points.parts():
                cv2.circle(frame,(point.x,point.y),2,(0,255,0),-1)

            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)


            if collect_count < count:
                now_time = time.time()

                if now_time - strat_time > interval:
                    face_descoriptor = face_descriptor_extractor.compute_face_descriptor(frame,points)
                    face_descoriptor = [i for i in face_descoriptor ]

                    line = [id,name,face_descoriptor]
                    csv_writer.writerow(line)

                    collect_count += 1
                    strat_time = now_time
                    # print('collect start!')
                    print('collect count: {collect_count}'.format(collect_count = collect_count))
                else:
                    pass
            else:
                print('collect finished!')
                return

        cv2.imshow("Face window",frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    f.close()        
    cap.release()
    cv2.destroyAllWindows()


def getFeaturelList():

    id_list = []
    name_list = []
    feature_list = None

    with open('C:\\Users\\Z16\\Desktop\\CV2\\Face attendance\\data\\face.csv','r') as f:
        csv_reader = csv.reader(f)

        for line in csv_reader:
            id = line[0]
            name = line[1]
            id_list.append(id)
            name_list.append(name)
            #string change to list
            face_descoriptor = eval(line[2])
            face_descoriptor = np.asarray(face_descoriptor,dtype=np.float64)
            face_descoriptor = np.reshape(face_descoriptor,(1,-1))

            if feature_list is None:
                feature_list = face_descoriptor
            else:
                feature_list = np.concatenate((feature_list,face_descoriptor),axis=0)
    return id_list,name_list,feature_list


def faceRecognizer(threshold=0.65):
    cap = cv2.VideoCapture(1)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hog_face_detector = dlib.get_frontal_face_detector()
    shape_detector = dlib.shape_predictor('C:\\Users\\Z16\\Desktop\\CV2\\shape_predictor_68_face_landmarks.dat')
    face_descriptor_extractor = dlib.face_recognition_model_v1('C:\\Users\\Z16\\Desktop\\CV2\\dlib_face_recognition_resnet_model_v1.dat')
    
    id_list,name_list,feature_list = getFeaturelList()

    recog_record = {}

    f = open('C:\\Users\\Z16\\Desktop\\CV2\\Face attendance\\data\\recognize.csv','a',newline="")
    csv_writer = csv.writer(f)

    while True:
        ret,frame = cap.read()

        frame = cv2.resize(frame,(width-50,height-50))

        frame = cv2.flip(frame,1)

        detections = hog_face_detector(frame,1)

        for face in detections:
            l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
            points = shape_detector(frame,face)

            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)

            face_descoriptor = face_descriptor_extractor.compute_face_descriptor(frame,points)
            face_descoriptor = [f for f in face_descoriptor ]

            face_descoriptor = np.asarray(face_descoriptor,dtype=np.float64)
            distances = np.linalg.norm((face_descoriptor-feature_list),axis=1)
            min_index = np.argmin(distances)
            min_distance = distances[min_index]

            if min_distance < threshold:
                predict_id = id_list[min_index]
                predict_name = name_list[min_index]
                now = time.time()
                need_save = False
                if predict_name in recog_record:
                    if now - recog_record[predict_name] > 3:
                        need_save = True
                        recog_record[predict_name] = now
                    else:
                        pass
                        need_save = False
                else:
                    need_save = True
                    recog_record[predict_name] = now

                if need_save:
                    time_local = time.localtime(recog_record[predict_name])
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
                    line = [predict_id,predict_name,min_distance,time_str]
                    csv_writer.writerow(line)
                    print("{time}: write successfully:{name}".format(name = predict_name,time = time_str))



            # else:
            #     print('unidentified')

        cv2.imshow("Face window",frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    f.close()
    cap.release()
    cv2.destroyAllWindows()


# faceRegister(id=1,name='paul',count=3,interval=3)

# id_list,name_list,feature_list = getFeaturelList()
# print(id_list)
# print(feature_list.shape)
# getFeaturelList()

faceRecognizer(threshold=0.65)