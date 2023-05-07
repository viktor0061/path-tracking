from bytetracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from supervision import Point
from supervision import VideoInfo
from supervision import get_video_frames_generator
from supervision import VideoSink
from supervision import Detections, BoxAnnotator
from supervision.detection.line_counter import LineZone, LineZoneAnnotator

from tqdm import tqdm

import cv2
from typing import List

import numpy as np

from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Input and output video path
SOURCE_VIDEO_PATH = "traffic-detection.mp4"
TARGET_VIDEO_PATH = "traffic-detection-result.mp4"

#Define BYTE parameters
#Detection score threshold
BYTETRACK_DETECTION_TRESH = 0.45
#Matching algorithm(i.e. IoU/Re-ID) threshold
BYTETRACK_MATCH_TRESH = 0.8
#Track buffer and frame rate will be used to determine the time after a lost track will be marked removed
BYTETRACK_TRACK_BUFFER = 30
BYTETRACK_FRAME_RATE = 30
    
#Converts Detections into format that can be consumed by ByteTracker.update()
def detections2boxes(detections: Detections) -> np.ndarray:
    
    return np.hstack((
        detections.xyxy,
        detections.confidence.reshape((-1, 1)),
        detections.class_id.reshape((-1, 1))
    ))

#Converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array(tracks[:, 0:4], dtype=float)


#Matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index][4]

    return tracker_ids

#Define the pre-trained model available ultralytics source repository
MODEL = "yolov8s.pt"

#Initialize detector instance
model = YOLO(MODEL)
model.fuse()

#Assign method mapping class_id to strings to a macro
CLASS_NAMES_DICT = model.model.names
#Class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]

#Create BYTETracker instance
byte_tracker = BYTETracker(track_thresh=BYTETRACK_DETECTION_TRESH, 
                           track_buffer=BYTETRACK_TRACK_BUFFER, 
                           match_thresh=BYTETRACK_MATCH_TRESH, 
                           frame_rate=BYTETRACK_FRAME_RATE)

#Create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
#Create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
#Create instance of BoxAnnotator
#box_annotator = BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

#Acquire first video frame
iterator = iter(generator)

#Initiate a progress bar
progress_bar = tqdm(total=video_info.total_frames, dynamic_ncols=True, position=0, leave=True)

with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    #initialise dictionary for tracks
    paths = {}

    #Loop over video frames
    for frame_number in range (video_info.total_frames):
        frame = next(iterator)
        #Detection on single frame
        results = model(frame)

        #Conversion to supervision detections
        detections = Detections.from_yolov8(results[0])

        #Filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        detection_array = detections2boxes(detections=detections)

        #Update tracks with detections
        tracks = byte_tracker.update(
            dets=detection_array, _=frame.shape
        )

        #Match the returned tracks with the input detection
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        #Filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        #Append new tracks to paths dictionary, assign a colour and add data points
        for xyxy, confidence, class_id, tracker_id in detections:
            if tracker_id not in paths:
                colour = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                paths[tracker_id] = {'colour':colour, 'points':[]}

            x, y = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2

            paths[tracker_id]['points'].append((x, y))

        #Draw the paths to the frame
        for path in paths.values():
            for point in path['points']:
                cv2.drawMarker(img=frame, 
                            position=(int(point[0]), int(point[1])), 
                            color=path['colour'], 
                            markerType=cv2.MARKER_STAR)

        #Format custom labels
        #labels = [
        #    f"{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        #    for _, confidence, class_id, tracker_id
        #    in detections
        #]

        #Annotate and display frame
        #frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        #Write frame to output
        sink.write_frame(frame)

        #update progress bar
        progress_bar.update(1)

        if cv2.waitKey(1)&0xFF==27:
            break

cv2.destroyAllWindows()
