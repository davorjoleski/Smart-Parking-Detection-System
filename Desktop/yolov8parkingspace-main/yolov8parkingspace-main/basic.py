import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        colorsBGR = [x, y]
        print(f"Clicked point: ({x}, {y})")

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('parking1.mp4')

if not cap.isOpened():
    print("ERROR: Cannot open video file")
    exit()

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

areas = {
    1: [(31,416), (73,417), (85,368), (53,364)],
    2: [(95,416), (134,415), (142,367), (104,366)],
    3: [(151,415), (196,413), (200,361), (161,365)],
    4: [(219,409), (267,406), (263,358), (215,359)],
    5: [(284,404), (339,401), (324,354), (277,356)],
    6: [(353,395), (404,395), (383,352), (338,350)],
    7: [(420,389), (471,387), (443,346), (400,348)],
    8: [(487,382), (534,383), (498,338), (461,343)],
    9: [(554,378), (590,375), (553,336), (517,338)],
    10: [(607,369), (643,365), (605,327), (573,330)],
    11: [(623,327), (660,360), (696,356), (650,323)],
    12: [(667,320), (713,354), (737,348), (694,316)],
    13: [(827,297), (868,320), (889,318), (842,292)],
    14: [(856,291), (899,314), (913,313), (872,291)],
    15: [(883,289), (923,310), (936,308), (895,287)],
    16: [(904,284), (944,305), (953,303), (915,283)],
    17: [(927,280), (964,299), (970,296), (934,278)]
}
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_fps = cap.get(cv2.CAP_PROP_FPS) or 30
output_width, output_height = 1020, 500
out = cv2.VideoWriter('parking_output.mp4', fourcc, output_fps, (output_width, output_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Сето останато оди тука
    frame = cv2.resize(frame, (1020,500))

    results = model.predict(frame, verbose=False)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    zone_counts = {k: 0 for k in areas.keys()}
    total_cars = 0

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            total_cars += 1
            cx = int((x1+x2)//2)
            cy = int((y1+y2)//2)

            for zone_id, polygon in areas.items():
                if cv2.pointPolygonTest(np.array(polygon, np.int32), (cx, cy), False) >= 0:
                    zone_counts[zone_id] += 1
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.circle(frame,(cx,cy),3,(0,0,255),-1)

    for zone_id, polygon in areas.items():
        if zone_counts[zone_id] > 0:
            color = (0,0,255)
            text = f"Spot {zone_id}: OCCUPIED ({zone_counts[zone_id]})"
            text_color = (0,0,255)
        else:
            color = (0,255,0)
            text = f"Zone {zone_id}: FREE"
            text_color = (0,255,0)

        cv2.polylines(frame, [np.array(polygon,np.int32)], True, color, 2)
        text_pos = (polygon[1][0], polygon[0][1]-30)
        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    cv2.putText(frame, f"Total Cars: {total_cars}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("RGB", frame)
    out.write(frame)
    if cv2.waitKey(1000) & 0xFF==27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
