import cv2
import mediapipe as mp
import triangulation_media_pipe as tmp
import numpy as np
import math
import os
import random


# get file list in path
def get_files(dir_path):
    res = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(dir_path + path)
    return res


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
change_face_list = get_files("./faces/")
previous_coordinates = []
JITTERING_THRESHOLD = 0.004
# mouth_landmark_index = [0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267]
mouth_landmark_index = [13, 82, 81, 80, 191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312]


MAIN_WINDOW_NAME = "Face Changer"

def load_base_img(face_mesh, image_file_name):
    image = cv2.imread(image_file_name)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return {"img": image, "landmarks": results}


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def normalized_to_pixel_coordinates(landmark_x, landmark_y, face_width, face_height):
    x = round_down(face_width * landmark_x)
    if x < 1:
        x = 1
    elif x >= face_width - 1:
        x = face_width - 2
    
    y = round_down(face_height * landmark_y)
    if y < 1:
        y = 1
    elif y >= face_height - 1:
        y = face_height - 2
    
    return (x, y)


def transform_landmarks_from_tf_to_ocv(keypoints, face_width, face_height):
    if not keypoints.multi_face_landmarks:
        return []
        
    landmark_list = []
    for face_landmarks in keypoints.multi_face_landmarks:
        for i in range(0, 468):
            landmark = face_landmarks.landmark[i]
            if abs(previous_coordinates[i][0] - landmark.x) < JITTERING_THRESHOLD:
                landmark.x = previous_coordinates[i][0]
            if abs(previous_coordinates[i][1] - landmark.y) < JITTERING_THRESHOLD:
                landmark.y = previous_coordinates[i][1]
            
            previous_coordinates[i] = (landmark.x, landmark.y)
            # pt = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, face_width, face_height)
            pt = normalized_to_pixel_coordinates(landmark.x, landmark.y, face_width, face_height)
            landmark_list.append(pt)
            
    return landmark_list

def get_mouth_coordinates_from_landmarks(landmarks):
    if not landmarks:
        return []
    mouth_coordinates = []
    for i in mouth_landmark_index:
        mouth_coordinates.append(landmarks[i])
    return mouth_coordinates


def match_triangles(keypoints, triangle_indexes):
    pass


def draw_triangulated_mesh(ocv_keypoints, img):
    for i in range(0, int(len(tmp.TRIANGULATION) / 3)):
        points = [tmp.TRIANGULATION[i * 3], tmp.TRIANGULATION[i * 3 + 1], tmp.TRIANGULATION[i * 3 + 2]]
        result1 = ocv_keypoints[points[0]]
        result2 = ocv_keypoints[points[1]]
        result3 = ocv_keypoints[points[2]]
        try:
            cv2.line(img, result1, result2, 255)
            cv2.line(img, result2, result3, 255)
            cv2.line(img, result3, result1, 255)
        except Exception:
            continue
    return img

def get_coordinates_array(list):
    result_array = np.empty()
    for element in list:
        result_array.append(np.int32(element))
    return result_array

def main():
    print("----------------shortcuts-----------")
    print("z -> toggle face landmarks")
    print("x -> change face_swaping base image randomly")
    print("c -> change face_swaping base image sequentially")
    print("S or s: save images")
    print("----------------shortcuts-----------")
    triangle_indexes = tmp.TRIANGULATION
    key_draw_landmarks = False
    key_draw_mask = False
    flag_change_face = False
    video_writer = None
    # face_list_index = random.randrange(0, len(change_face_list))
    face_list_index = 0
    
    # For webcam input:
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    for i in range(0, 468):
        previous_coordinates.append((0.0, 0.0))
    
    base_face_handler, landmark_base_ocv, base_input_image = process_base_face_mesh(drawing_spec, face_mesh, change_face_list[face_list_index], show_landmarks=key_draw_landmarks, show_triangulated_mesh=key_draw_mask)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, webcam_img = cap.read()
        if not success:
            break
            
        if flag_change_face:
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            base_face_handler, landmark_base_ocv, base_input_image = process_base_face_mesh(drawing_spec, face_mesh, change_face_list[face_list_index], show_landmarks=key_draw_landmarks, show_triangulated_mesh=key_draw_mask)
            flag_change_face = False
        image_rows, image_cols, _ = webcam_img.shape
        webcam_img.flags.writeable = False
        results = face_mesh.process(webcam_img)
        landmark_target_ocv = transform_landmarks_from_tf_to_ocv(results, image_cols, image_rows)
        mouth_coordinates = get_mouth_coordinates_from_landmarks(landmarks=landmark_target_ocv)
        
        # Draw the face mesh annotations on the image.
        webcam_img.flags.writeable = True
        image = webcam_img.copy()
        seam_clone = image.copy()
        result = webcam_img.copy()
        out_image = webcam_img.copy()
        img2_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(image)
        seamlessclone = webcam_img.copy()
        if not results.multi_face_landmarks:
            continue
        
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=out_image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)

            # out_image = draw_triangulated_mesh(landmark_target_ocv, webcam_img)
            if len(landmark_target_ocv) > 0:
                points2 = np.array(landmark_target_ocv, dtype=np.int32)
                mouth_points = np.array(mouth_coordinates, dtype=np.int32)
                
                convexhull2 = cv2.convexHull(points2)
                convexhull_mouth = cv2.convexHull(mouth_points)
                
                for i in range(0, int(len(tmp.TRIANGULATION) / 3)):
                    triangle_index = [tmp.TRIANGULATION[i * 3],
                                      tmp.TRIANGULATION[i * 3 + 1],
                                      tmp.TRIANGULATION[i * 3 + 2]]
                    tbas1 = landmark_base_ocv[triangle_index[0]]
                    tbas2 = landmark_base_ocv[triangle_index[1]]
                    tbas3 = landmark_base_ocv[triangle_index[2]]
                    triangle1 = np.array([tbas1, tbas2, tbas3], np.int32)

                    rect1 = cv2.boundingRect(triangle1)
                    (x, y, w, h) = rect1
                    x = x - 1
                    y = y - 1
                    w = w + 1
                    h = h + 1
                    cropped_triangle = base_input_image[y: y + h, x: x + w]
                    cropped_tr1_mask = np.zeros((h, w), np.uint8)

                    points = np.array([[tbas1[0] - x, tbas1[1] - y],
                                       [tbas2[0] - x, tbas2[1] - y],
                                       [tbas3[0] - x, tbas3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
                    ttar1 = landmark_target_ocv[triangle_index[0]]
                    ttar2 = landmark_target_ocv[triangle_index[1]]
                    ttar3 = landmark_target_ocv[triangle_index[2]]
                    triangle2 = np.array([ttar1, ttar2, ttar3], np.int32)

                    rect2 = cv2.boundingRect(triangle2)
                    (x, y, w, h) = rect2
                    x = x - 1
                    y = y - 1
                    w = w + 1
                    h = h + 1
                    
                    cropped_tr2_mask = np.zeros((h, w), np.uint8)

                    points2 = np.array([[ttar1[0] - x, ttar1[1] - y],
                                        [ttar2[0] - x, ttar2[1] - y],
                                        [ttar3[0] - x, ttar3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
                    
                    # Warp triangles
                    points = np.float32(points)
                    points2 = np.float32(points2)
                    M = cv2.getAffineTransform(points, points2)
                    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                    # Reconstructing destination face
                    img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
                    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
                    
                # Face swapped (putting 1st face into 2nd face)
                img2_face_mask = np.zeros_like(img2_gray)
                img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, (255, 255, 255), cv2.LINE_8, 0)
                img2_face_mask = cv2.bitwise_not(img2_head_mask)
                
                img2_head_noface = cv2.bitwise_and(seam_clone, seam_clone, mask=img2_face_mask)
                result = cv2.add(img2_head_noface, img2_new_face)
                
                (x, y, w, h) = cv2.boundingRect(convexhull2)
                center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
                seamlessclone = cv2.seamlessClone(result, seam_clone, img2_head_mask, center_face2, cv2.MIXED_CLONE)
                
                # Extract mouth part
                img2_mouth = np.zeros_like(img2_gray)
                img2_mouth = cv2.fillConvexPoly(img2_mouth, convexhull_mouth, (255, 255, 255), cv2.LINE_8, 0)

                mouth_img = cv2.bitwise_and(seam_clone, seam_clone, mask=img2_mouth)
                
                img2_mouth = cv2.bitwise_not(img2_mouth)
                seamlessclone = cv2.bitwise_and(seamlessclone, seamlessclone, mask=img2_mouth)
                seamlessclone = cv2.add(seamlessclone, mouth_img)
                
        cv2.imshow('Input Image', base_face_handler["img"])
        # cv2.imshow('MediaPipe FaceMesh Landmarks', out_image)
        # cv2.imshow('Face Swap Result', result)
        cv2.imshow(MAIN_WINDOW_NAME, seamlessclone)
        key = cv2.waitKey(5)
        if key == 27:
            break;
        if key == 122:
            key_draw_landmarks = not key_draw_landmarks
        if key == 120:
            flag_change_face = not flag_change_face
            new_index = random.randrange(0, len(change_face_list))
            while face_list_index == new_index:
                new_index = random.randrange(0, len(change_face_list))
            face_list_index = new_index
        if key == 99:
            flag_change_face = not flag_change_face
            face_list_index = face_list_index + 1
            if face_list_index >= len(change_face_list):
                face_list_index = 0
    face_mesh.close()
    cap.release()
    video_writer.release()

def process_base_face_mesh(drawing_spec, face_mesh, image_file, show_landmarks=False, show_triangulated_mesh=False):
    base_face_handler = load_base_img(face_mesh, image_file)
    base_input_image = base_face_handler["img"].copy()
    image_rows, image_cols, _ = base_face_handler["img"].shape
    landmark_base_ocv = transform_landmarks_from_tf_to_ocv(base_face_handler["landmarks"], image_cols, image_rows)
    if show_landmarks:
        mp_drawing.draw_landmarks(
            image=base_face_handler["img"],
            landmark_list=base_face_handler["landmarks"].multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    if show_triangulated_mesh:
        base_face_handler["img"] = draw_triangulated_mesh(landmark_base_ocv, base_face_handler["img"])
    
    return base_face_handler, landmark_base_ocv, base_input_image


if __name__ == "__main__":
    main()
