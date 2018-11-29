# -*- coding: UTF-8 -*-

import cv2 as cv
import numpy as np

class CVFaceDetector(object):
	"""docstring for CVFaceDetector"""
	''' 
	proto_path: prototxt path 
	weights_path: model weights
	'''
	def __init__(self, proto_path, weights_path, det_conf_thresh = 0.9,
				 nms_thresh=0.45, scale=1.0, mean=[0,0,0], is_rgb=True ):
		if proto_path is None or weights_path is None:
			raise Error("caffe model & prototxt are none" )

		self.net = cv.dnn.readNetFromCaffe( prototxt=proto_path, caffeModel=weights_path )
		self.nms_thresh = nms_thresh
		self.det_thresh = det_conf_thresh
		self.scale = scale
		self.mean = mean
		self.rgb = is_rgb

	def _getOutputsNames(self):
		layersNames = self.net.getLayerNames()
		return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

	def detect_face( self, img ):
		if img is None:
			return []
		
		inpHeight = img.shape[0]
		inpWidth = img.shape[1]

		blob = cv.dnn.blobFromImage( img, self.scale, (inpWidth, inpHeight), self.mean, self.rgb, crop=False)
		self.net.setInput( blob )
		outs = self.net.forward( self._getOutputsNames() )

		return self.postprocess( img, outs );

	def postprocess(self, frame, outs):
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
	
		layerNames = self.net.getLayerNames()
		lastLayerId = self.net.getLayerId(layerNames[-1])
		lastLayer = self.net.getLayer(lastLayerId)
	
		classIds = []
		confidences = []
		boxes = []
	
		results = []
	
		for out in outs:
			for detection in out[0, 0]:
				confidence = detection[2]
				if confidence > self.det_thresh:
					left = int(detection[3] * frameWidth)
					top = int(detection[4] * frameHeight)
					right = int(detection[5] * frameWidth)
					bottom = int(detection[6] * frameHeight)
					width = right - left + 1
					height = bottom - top + 1
					classIds.append(int(detection[1]) - 1)  # Skip background label
					confidences.append(float(confidence))
					boxes.append([left, top, width, height])
	   	'''                 
	    if self.net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
	        # Network produces output blob with a shape 1x1xNx7 where N is a number of
	        # detections and an every detection is a vector of values
	        # [batchId, classId, confidence, left, top, right, bottom]
	        for out in outs:
	            for detection in out[0, 0]:
	                confidence = detection[2]
	                if confidence > self.det_thresh:
	                    left = int(detection[3])
	                    top = int(detection[4])
	                    right = int(detection[5])
	                    bottom = int(detection[6])
	                    width = right - left + 1
	                    height = bottom - top + 1
	                    classIds.append(int(detection[1]) - 1)  # Skip background label
	                    confidences.append(float(confidence))
	                    boxes.append([left, top, width, height])
	    elif lastLayer.type == 'DetectionOutput':
	        # Network produces output blob with a shape 1x1xNx7 where N is a number of
	        # detections and an every detection is a vector of values
	        # [batchId, classId, confidence, left, top, right, bottom]
	        for out in outs:
	            for detection in out[0, 0]:
	                confidence = detection[2]
	                if confidence > self.det_thresh:
	                    left = int(detection[3] * frameWidth)
	                    top = int(detection[4] * frameHeight)
	                    right = int(detection[5] * frameWidth)
	                    bottom = int(detection[6] * frameHeight)
	                    width = right - left + 1
	                    height = bottom - top + 1
	                    classIds.append(int(detection[1]) - 1)  # Skip background label
	                    confidences.append(float(confidence))
	                    boxes.append([left, top, width, height])
	    elif lastLayer.type == 'Region':
	        # Network produces output blob with a shape NxC where N is a number of
	        # detected objects and C is a number of classes + 4 where the first 4
	        # numbers are [center_x, center_y, width, height]
	        classIds = []
	        confidences = []
	        boxes = []
	        for out in outs:
	            for detection in out:
	                scores = detection[5:]
	                classId = np.argmax(scores)
	                confidence = scores[classId]
	                if confidence > self.det_thresh:
	                    center_x = int(detection[0] * frameWidth)
	                    center_y = int(detection[1] * frameHeight)
	                    width = int(detection[2] * frameWidth)
	                    height = int(detection[3] * frameHeight)
	                    left = int(center_x - width / 2)
	                    top = int(center_y - height / 2)
	                    classIds.append(classId)
	                    confidences.append(float(confidence))
	                    boxes.append([left, top, width, height])
	    else:
	        print('Unknown output layer type: ' + lastLayer.type)
	        exit()
		'''

		indices = cv.dnn.NMSBoxes(boxes, confidences, self.det_thresh, self.nms_thresh )
		for i in indices:
		    i = i[0]
		    box = boxes[i]
		    left = box[0]
		    top = box[1]
		    width = box[2]
		    height = box[3]
		    results.append( [confidences[i], left, top, width, height] )
		
		    #drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
		return results