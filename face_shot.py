import cv2

# Your existing code to capture and save images with the received name
name = "Hop"
cam = cv2.VideoCapture(0)
data1 = ''
cv2.namedWindow("Set for " + name, cv2.WINDOW_NORMAL)
cv2.resizeWindow("Set for " + name, 500, 300)

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Set for " + name, frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        print("Escape hit, closing..")
        break
    elif k%256 == 32:
        img_name = "Data/{}/image_{}.jpg".format(name, img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
