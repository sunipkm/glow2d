# %%
import numpy as np
import cv2
img = np.uint8(np.random.rand(8, 8)*255)
#array([[230,  45, 153, 233, 172, 153,  46,  29],
#       [172, 209, 186,  30, 197,  30, 251, 200],
#       [175, 253, 207,  71, 252,  60, 155, 124],
#       [114, 154, 121, 153, 159, 224, 146,  61],
#       [  6, 251, 253, 123, 200, 230,  36,  85],
#       [ 10, 215,  38,   5, 119,  87,   8, 249],
#       [  2,   2, 242, 119, 114,  98, 182, 219],
#       [168,  91, 224,  73, 159,  55, 254, 214]], dtype=uint8)

map_y = np.array([[0, 1], [2, 3]], dtype=np.float32)
map_x = np.array([[5, 6], [7, 10]], dtype=np.float32)
mapped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
# %%
import matplotlib.pyplot as plt

plt.imshow(img)
plt.show()
plt.imshow(mapped_img)
plt.show()
# %%
cv2.polarToCart()