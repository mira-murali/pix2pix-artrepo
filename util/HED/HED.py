# USAGE
# python detect_edges_image.py --edge-detector hed_model --image images/guitar.jpg

# import the necessary packages
import cv2, time, random, sys, gc
import os
from tqdm import tqdm
from multiprocessing import Pool


class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

def HED(image_file):
	if ".png" not in image_file:
		return
	# load our serialized edge detector from disk
	protoPath = os.path.sep.join(["/home/loki/art/colab/holistically-nested-edge-detection/hed_model",
		"deploy.prototxt"])
	modelPath = os.path.sep.join(["/home/loki/art/colab/holistically-nested-edge-detection/hed_model",
		"hed_pretrained_bsds.caffemodel"])
	net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


	# load the input image and grab its dimensions
	image = cv2.imread(image_file)
	(H, W) = image.shape[:2]

	# construct a blob out of the input image for the Holistically-Nested
	# Edge Detector
	blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
		mean=(104.00698793, 116.66876762, 122.67891434),
		swapRB=False, crop=False)

	# set the blob as the input to the network and perform a forward pass
	# to compute the edges
	net.setInput(blob)
	hed = net.forward()
	hed = cv2.resize(hed[0, 0], (W, H))
	hed = (255 * hed).astype("uint8")

	# show the output edge detection results for Canny and
	# Holistically-Nested Edge Detection
	cv2.imwrite("hed/"+image_file, hed)
	del net, image, blob, hed
	return

if __name__ == "__main__":
	imgList = os.listdir(".")
	imgList.sort()
	
	# register our new layer with the model
	cv2.dnn_registerLayer("Crop", CropLayer)



	threads = 5
	chunking = 10
	p = Pool(threads)
	#gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
	fh = open("../save512.txt",'r')
	start = int(fh.readlines()[0])
	fh.close()
	for n,i in enumerate(tqdm(range(start,len(imgList),threads*chunking))):
		if n>10:
			sys.exit()
		if n%3==0:
			gc.collect()
		p.map(HED, imgList[i:i+threads*chunking])
		with open("../save512.txt",'w') as save:
			save.writelines([str(i+threads*chunking)])
