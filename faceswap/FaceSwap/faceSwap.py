import os
import dlib
import cv2
import numpy as np
import models
import NonLinearLeastSquares
import ImageProcessing
from drawing import *
import FaceRendering
import utils


predictor_path = os.path.join(os.path.dirname(__file__), "..", "shape_predictor_68_face_landmarks.dat")
image_name = os.path.join(os.path.dirname(__file__), "..", "data", "Chaitra.jpeg")
maxImageSizeForDetection = 1080

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel( os.path.join(os.path.dirname(__file__), "..", "candide.npz"))

projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])


drawOverlay = False
cap = cv2.VideoCapture("female3Dvideo.mp4")
writer = None
cameraImg = cap.read()[1]
print(cameraImg)
textureImg = cv2.imread(image_name)
textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)

while True:
    cameraImg = cap.read()[1]
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

    if shapes2D is not None:
        for shape2D in shapes2D:
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

            shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D)

            mask = np.copy(renderedImg[:, :, 0])
            renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
            cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)


            if drawOverlay:
                drawPoints(cameraImg, shape2D.T)
                drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)

    if writer is not None:
        writer.write(cameraImg)
        writer = cv2.VideoWriter(os.path.join(os.path.dirname(__file__), "..", "out.avi"),
                                 cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                 25,
                                 (cameraImg.shape[1], cameraImg.shape[0]))


    cv2.imshow('image', cameraImg)
    key = cv2.waitKey(1)

    if key == 27:
        break
    if key == ord('t'):
        drawOverlay = not drawOverlay
    if key == ord('r'):
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(os.path.dirname(__file__), "..", "out.avi"),
                                     cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                     25,
                                     (cameraImg.shape[1], cameraImg.shape[0]))

            if writer.isOpened():
                print("Writer succesfully opened")
            else:
                writer = None
                print("Writer opening failed")
        else:
            print("Stopping video writer")
            writer.release()
            writer = None
