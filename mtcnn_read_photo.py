import tensorflow as tf
import cv2
import align.detect_face
import numpy as np

minsize = 20  # 脸部最小尺寸
threshold = [0.6, 0.7, 0.7]  # 三步阈值
factor = 0.709  # 缩放金字塔缩放因子
dist = []
name_tmp = []
Emb_data = []
image_tmp = []
img = 'faceset/nana.jpg'
img_color = 'red'


def read_photo():
    print(' 建立mtcnn人脸检测模型，加载参数')
    gpu_memory_fraction = 1.0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, 'align/')

    frame = cv2.imread(img)
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

    cv2.imshow('src', frame)
    bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    faces_num = bounding_boxes.shape[0]  # 人脸数目
    print('bounding_boxes:', bounding_boxes)
    print('_:', _)
    print('找到人脸数目为：{}'.format(faces_num))
    print(bounding_boxes.shape)
    print(bounding_boxes)
    print(np.squeeze(bounding_boxes))
    print(np.squeeze(bounding_boxes).shape)

    crop_faces = []
    for face_position in bounding_boxes:
        face_position = face_position.astype(int)
        # print(face_position[0:4])
        cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 1)

    cv2.imshow('demo', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    read_photo()
