import tensorflow as tf
import cv2
import align.detect_face
import os
import numpy as np

minsize = 20  # 脸部最小尺寸
threshold = [0.6, 0.7, 0.7]  # 三步阈值
factor = 0.709  # 缩放金字塔缩放因子
dist = []
name_tmp = []
Emb_data = []
image_tmp = []
img_path = 'faceset/me.jpg'


# 获取最大人脸索引
def max_face(area, position):
    empty = False
    max_face_position = []
    if not area:
        empty = True
    else:
        max_area_index = np.argmax(area)
        print('最大面积索引：', np.argmax(area), '最大面积：', max(area))
        max_face_position = position[max_area_index]
    return max_face_position, empty


def read_photo(img):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, 'align/')

    frame = cv2.imread(img)
    # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    cv2.imshow('src', frame)
    bounding_boxes, points = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    faces_num = bounding_boxes.shape[0]  # 人脸数目
    print('bounding_boxes.shape:', bounding_boxes.shape, '\n bounding_boxes:', bounding_boxes)
    # print('points shape:', points.shape, '\n points:', points)
    print('找到人脸数目为：{}'.format(faces_num))

    Index = []  # 序列
    Area = []  # 面积
    Position = []  # 坐标

    for i, face_position in enumerate(bounding_boxes):
        face_position = face_position.astype(int)
        w = face_position[2] - face_position[0]
        h = face_position[3] - face_position[1]
        S = w * h
        print('w:', face_position[2], '-', face_position[0], '=', w)
        print('h', face_position[3], '-', face_position[1], '=', h, '\n')
        print('-->', i + 1)
        Index.append(i)
        Area.append(S)
        Position.append(face_position)

        max_face_position, is_empty = max_face(Area, Position)
        # 如果不是空的绘制面部边框
        if is_empty is False:
            cv2.rectangle(frame, (max_face_position[0], max_face_position[1]),
                          (max_face_position[2], max_face_position[3]),
                          (0, 255, 0), 1)
            cv2.circle(frame, (max_face_position[0], max_face_position[1]), 2, (0, 0, 255), -1)
            cv2.circle(frame, (max_face_position[2], max_face_position[3]), 2, (0, 0, 255), -1)
            cv2.putText(
                frame,
                'Max Face',
                (max_face_position[0], max_face_position[1]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 0, 255),
                thickness=1,
                lineType=1)

    # writer = tf.summary.FileWriter('logs/', sess.graph)
    cv2.imshow('demo', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_photo(img_path)
