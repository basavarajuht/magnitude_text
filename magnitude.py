import cv2
from fcmeans import FCM
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.cluster import KMeans

def get_magnitude(img):
    st = time()
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    et = time()

    print(f'Time taken for calculating magnitude {et-st} seconds')

    return magnitude

img_path = 'C:\\Users\\Lenovo\\Desktop\\17.png'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

magnitude = get_magnitude(img)
a=magnitude.shape
I1 = magnitude.reshape((-1, 1))
fcm = FCM(n_clusters=2)
fcm.fit(I1)
fcm_centers = fcm.centers
fcm_labels = fcm.predict(I1)
reshaped1 = fcm_labels.reshape((a[0],a[1]))
res1 = np.where(reshaped1 == 1)
res0 = np.where(reshaped1 == 0)
result = np.zeros((magnitude.shape), np.uint8)
len1=len(res1[0])
len0=len(res0[0])
if len1<len0:
    result[res1[0], res1[1]] = 255
else:
    result[res0[0], res0[1]] = 255
se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
final_output = cv2.dilate(result, se1)
plt.figure()
plt.imshow(final_output,cmap='gray')
#plt.imshow(magnitude, cmap='gray')
#plt.imshow(kmeans, cmap='gray')
plt.show()
cv2.imwrite('fuzzy_17.jpg', final_output)

