from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

model = YOLO("best.pt")
names = model.names
trajectory = defaultdict(list)

results = model.track(
    source="video/input_video.mp4",
    persist=True,
    tracker="bytetrack.yaml",
    conf=0.25,
    stream=True,
    save=False,
)

for frame_idx, r in enumerate(results):
    if r.boxes.id is None:
        continue
    ids = r.boxes.id.int().tolist()
    clss = r.boxes.cls.int().tolist()
    xywh = r.boxes.xywh.tolist()
    for tid, c, (x, y, w, h) in zip(ids, clss, xywh):
        trajectory[tid].append((frame_idx, names[c], x, y))

rows = [
    {"id": tid, "class": cls_name, "frame": f, "x": x, "y": y}
    for tid, pts in trajectory.items()
    for f, cls_name, x, y in pts
]
df = pd.DataFrame(rows)
df.to_csv("trajectory.csv", index=False)

plt.figure(figsize=(9, 6))
for (tid, cls_name), g in df.groupby(["id", "class"]):
    if len(g) < 10:
        continue
    plt.plot(g["x"], g["y"], marker=".", markersize=2, label=f"{cls_name} id{tid}")
plt.gca().invert_yaxis()
plt.xlabel("x (pixel)")
plt.ylabel("y (pixel)")
plt.title("Object Trajectories")
plt.legend(fontsize=8)
plt.savefig("trajectory.png", dpi=150, bbox_inches="tight")
