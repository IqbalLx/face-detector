import cv2

from face_detector import DetectFace

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)

    while True:
        is_frame, frame = cam.read()

        if not is_frame:
            break

        face_detector = DetectFace(frame).using_haar()
        is_face, face_coords = face_detector.extract()
        if is_face:
            for x1, y1, x2, y2 in face_coords:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow('Preview', frame)
        if cv2.waitKey(1) & 0xff == ord('x'):
            break

    cam.release()
    cv2.destroyAllWindows()