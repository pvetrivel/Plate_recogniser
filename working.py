import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

#def canny(image):
#    gray=cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
#   blur=cv2.GaussianBlur(gray,(5,5),0)
#    canny=cv2.Canny(blur,50,150)
#    return canny
#read image
image=cv2.imread('car.jpg')
lane_image=np.copy(image)
#convert to grey
lane_image=cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
#canny=canny(lane_image)
#crop=canny[252:304,295:450] 
#remove noise
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(lane_image, kernel, iterations=1)
#write after removed noise
cv2.imwrite("removed_noise.png", img)
result = pytesseract.image_to_string("removed_noise.png")
print(pytesseract.image_to_string("thres.png"))
print(result)


#plt.imshow(img)
#plt.imshow(edged)
#plt.show()