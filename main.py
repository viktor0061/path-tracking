import ultralytics
import sys
from bytetracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
import supervision
from supervision import ColorPalette
from supervision import Point
from supervision import VideoInfo
from supervision import get_video_frames_generator
from supervision import VideoSink
from supervision import show_frame_in_notebook
from supervision import Detections, BoxAnnotator
from supervision.detection.line_counter import LineZone, LineZoneAnnotator

import cv2
from typing import List

import numpy as np

from ultralytics import YOLO
import matplotlib.pyplot as plt


SOURCE_VIDEO_PATH = "traffic-detection.mp4"

BYTETRACKER_TRACK_TRESH = 0.25
BYTETRACKER_TRACK_BUFFER = 30
BYTETRACKER_MATCH_TRESH = 0.4
BYTETRACKER_FRAME_RATE = 15
    
# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    
    return np.hstack((
        detections.xyxy,
        detections.confidence.reshape((-1, 1)),
        detections.class_id.reshape((-1, 1))
    ))

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array(tracks[:, 0:4], dtype=float)


# matches our bounding boxes with predictions
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

MODEL = "yolov8s.pt"

model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]


# settings
LINE_START = Point(50, 1500)
LINE_END = Point(3840-50, 1500)

TARGET_VIDEO_PATH = "traffic-detection-result.mp4"

# create BYTETracker instance
byte_tracker = BYTETracker(track_thresh=BYTETRACKER_TRACK_TRESH, 
                           track_buffer=BYTETRACKER_TRACK_BUFFER, 
                           match_thresh=BYTETRACKER_MATCH_TRESH, 
                           frame_rate=BYTETRACKER_FRAME_RATE)

# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create LineCounter instance
line_counter = LineZone(start=LINE_START, end=LINE_END)
# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
line_annotator = LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

# acquire first video frame
iterator = iter(generator)

with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # loop over video frames
    for frame_number in range (video_info.total_frames):

        frame = next(iterator)
        # model prediction on single frame and conversion to supervision Detections
        results = model(frame)

        detections = Detections.from_yolov8(results[0])

        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # tracking detections
        tracks = byte_tracker.update(
            dets=detections2boxes(detections=detections), _=frame.shape
        )

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # format custom labels
        labels = [
            f"{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        # updating line counter
        line_counter.trigger(detections=detections)
        #print(line_counter.in_count)
        #print(line_counter.out_count)

        # annotate and display frame
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        sink.write_frame(frame)
        #cv2.imshow("IMG",frame)
        if cv2.waitKey(1)&0xFF==27:
            break

cv2.destroyAllWindows()
