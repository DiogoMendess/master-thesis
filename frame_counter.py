import cv2

cap = cv2.VideoCapture("C:/Users/diogo/Desktop/Tese/Dados/Videos/14.03.2022/20220314_1.03_blurred/20220314_1.03_1_9_blurred.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )