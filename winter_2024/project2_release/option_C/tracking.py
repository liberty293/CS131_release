import cv2
import numpy as np

tracker_types = [ 'Boost', 'MIL','CSRT', 'TLD', 'MedFlow', 'Mosse']



def getTracker(tracker_type):
    match tracker_type:
        case 'Boost':
            tracker = cv2.legacy_TrackerBoosting.create()
        case 'MIL':
            tracker = cv2.TrackerMIL_create()
        case 'KCF':
            tracker = cv2.TrackerKCF_create()
        case 'CSRT':
            tracker = cv2.legacy_TrackerCSRT.create()
        case 'TLD':
            tracker = cv2.legacy_TrackerTLD.create()
        case 'MedFlow':
            tracker = cv2.legacy_TrackerMedianFlow.create()
        case 'Mosse':
            tracker = cv2.legacy_TrackerMOSSE.create()
    return tracker



#if webcam
video = cv2.VideoCapture(0)

#for video
import os
# base_folder = os.path.dirname(__file__)
# image_path = os.path.join(base_folder, 'vid.mp4')
#print("file exists?", os.path.exists(image_path))
#video = cv2.VideoCapture(image_path)
#frame_cnt = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#ok, frame = video.read()

#if choose box
while True:
    k,frame = video.read()
    cv2.imshow("Tracking",frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
bbox = cv2.selectROI(frame, False)

#for set box
# from utils import load_bboxes, animated_bbox, load_frames
# gt_bboxes = load_bboxes(os.path.join(base_folder, 'Man/groundtruth_rect.txt'))



#if choose box
tracker = getTracker(tracker_types[5])
ok = tracker.init(frame, bbox)
cv2.destroyWindow("ROI selector")
keypnts = []
#if webcam
while True:
    ok, frame = video.read()
    ok, bbox = tracker.update(frame)
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]),
              int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0,0,255), 2, 2)
    cv2.imshow("Tracking", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

#if video
# from motion import IoU
# #start timer
# #for tracker_type in tracker_types:

# mytracker = getTracker(tracker_types[4])
# print(mytracker)
# x, y, w, h = gt_bboxes[0]
# bbox = [(x, y, w, h)]
# bbox_current = (x, y, w, h)
# timer = cv2.getTickCount()
# ok, frame = video.read()
# # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
# ok = mytracker.init(frame, bbox_current)
# print(ok)
# while True:
#     ok, frame = video.read()
#     if not ok:
#         break
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#     ok, bbox_current = mytracker.update(frame)
#     #draw box
#     if ok:
#         x, y, w, h = bbox_current
#         bbox.append((x, y, w, h))
#     else:
#         bbox.append((0,0,0,0))


# fps_proc = (cv2.getTickCount() - timer) /(cv2.getTickFrequency()*frame_cnt)
# print(tracker_types[0] + " fps "+ str(fps_proc))


# average_iou = 0.0
# for gt_bbox, bboxes in zip(gt_bboxes, bbox):
#     average_iou += IoU(gt_bbox, bboxes)
    
# average_iou /= len(gt_bboxes)
# print(tracker_types[0] + " iou "+ str(average_iou))

# frames = load_frames(os.path.join(base_folder, 'Man/img'))

# ani = animated_bbox(frames, bbox)
# ani.save(os.path.join(base_folder, 'test.mp4'), fps=16.67)

# def pyramidal_horn_schunck(im1, im2, alpha=1.0, iterations=100, epsilon=0.01):
#     # Compute derivatives of images
#     fx = np.gradient(im1, axis=1)
#     fy = np.gradient(im1, axis=0)
#     ft = im2 - im1

#     # Initial flow vectors
#     u = np.zeros_like(im1)
#     v = np.zeros_like(im1)

#     for _ in range(iterations):
#         # Compute local averages of the flow vectors
#         u_avg = np.convolve(u, np.ones((3, 3)) / 9, mode='same')
#         v_avg = np.convolve(v, np.ones((3, 3)) / 9, mode='same')

#         # Compute flow vectors incrementally
#         du = ((fx * fx) * u_avg + fx * fy * v_avg + fx * ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
#         dv = ((fy * fy) * v_avg + fx * fy * u_avg + fy * ft) / (alpha ** 2 + fx ** 2 + fy ** 2)

#         # Update flow vectors
#         u = u_avg - du
#         v = v_avg - dv

#         # Convergence check
#         if np.sum(du ** 2 + dv ** 2) < epsilon:
#             break

#     return u, v