import numpy as np
import cv2

import ImageProcessing

eyeImg = cv2.imread("../data/eye.png")
maskImg = cv2.imread("../data/mask.png")

mask = np.mean(maskImg, axis=2)

eyeImg = ImageProcessing.colorTransfer(eyeImg, mask)
blendedImg = ImageProcessing.blendImages(eyeImg, mask)

cv2.imwrite(blendedImg)
