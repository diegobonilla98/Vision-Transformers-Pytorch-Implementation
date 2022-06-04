import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TKAgg')

image = cv2.imread('test.jpg')[:, :, ::-1]  # R(HxWxC) -> R(Nx(P²*C))
image = cv2.resize(image, (512, 512))

N = 4
P = int(image.shape[0] / np.sqrt(N))
C = image.shape[2]



# cv2.imshow("Image", image)
# cv2.waitKey()

