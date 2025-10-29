import cv2

cap = cv2.VideoCapture('/dev/video1')
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    #bgr = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
    cv2.imshow("test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):  # 'q' or ESC
        break


cap.release()
cv2.destroyAllWindows()