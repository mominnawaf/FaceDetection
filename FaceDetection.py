import cv2
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
face_cascead = cv2.CascadeClassifier(
    "\\Dep\\haarcascade_frontalface_default.xml")
eye_casecade = cv2.CascadeClassifier(
    "\\Dep\\haarcascade_eye.xml")
video = cv2.VideoCapture(0)
while True:
    check, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascead.detectMultiScale(gray, 1.05, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_casecade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 4)
        roi_smile_gray = gray[y:y + h, x:x + w]
        roi_smile_col = img[y:y + h, x:x + w]
        smile=smile_casecade.detectMultiScale(roi_smile_gray,minNeighbors=15, minSize=(25, 25))
        for sx,sy,sw,sh in smile:
            cv2.rectangle(roi_smile_col, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 4)
    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
