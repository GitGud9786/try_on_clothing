import json, math
from pathlib import Path

# Adjust if your MediaPipe JSON structure differs.
# Expecting: {"people":[{"pose_keypoints_2d":[x0,y0,v0, x1,y1,v1, ...]}]}
# If your files are in a different schema, tweak `read_mediapipe_triplets` accordingly.

SRC = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test\pose")              # folder with your current MediaPipe-style JSONs
DST = Path(r"D:\CSE\Thesis\try_on_clothing\test_set\my_dataset\test\pose_openpose")     # output folder with COCO-18 OpenPose-style JSONs
DST.mkdir(exist_ok=True, parents=True)

# MediaPipe indices we’ll sample
MP = {
    "nose": 0, "l_eye": 2, "r_eye": 5, "l_ear": 7, "r_ear": 8,
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28,
}

def read_mediapipe_triplets(fp):
    with open(fp, "r") as f:
        data = json.load(f)
    # Your earlier code saved under "people[0].pose_keypoints_2d" already as [x,y,v] in pixels.
    # If your file is different, adapt here.
    pts = data["people"][0]["pose_keypoints_2d"]
    # make list of (x,y,v)
    return [(pts[i], pts[i+1], pts[i+2]) for i in range(0, len(pts), 3)]

def midpoint(a, b):
    return ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0)

for jf in sorted(SRC.glob("*_keypoints.json")):
    mp_pts = read_mediapipe_triplets(jf)

    # helper to grab a mediapipe joint safely
    def J(idx): return mp_pts[idx] if 0 <= idx < len(mp_pts) else (0.0, 0.0, 0.0)

    # build COCO-18 in order
    nose        = J(MP["nose"])
    neck        = midpoint(J(MP["l_shoulder"]), J(MP["r_shoulder"]))
    r_shoulder  = J(MP["r_shoulder"])
    r_elbow     = J(MP["r_elbow"])
    r_wrist     = J(MP["r_wrist"])
    l_shoulder  = J(MP["l_shoulder"])
    l_elbow     = J(MP["l_elbow"])
    l_wrist     = J(MP["l_wrist"])
    r_hip       = J(MP["r_hip"])
    r_knee      = J(MP["r_knee"])
    r_ankle     = J(MP["r_ankle"])
    l_hip       = J(MP["l_hip"])
    l_knee      = J(MP["l_knee"])
    l_ankle     = J(MP["l_ankle"])
    r_eye       = J(MP["r_eye"])
    l_eye       = J(MP["l_eye"])
    r_ear       = J(MP["r_ear"])
    l_ear       = J(MP["l_ear"])

    coco18 = [nose, neck, r_shoulder, r_elbow, r_wrist,
              l_shoulder, l_elbow, l_wrist, r_hip, r_knee, r_ankle,
              l_hip, l_knee, l_ankle, r_eye, l_eye, r_ear, l_ear]

    flat = []
    for (x,y,v) in coco18:
        flat += [float(x), float(y), float(v)]

    out = {"version": 1.2, "people": [{"pose_keypoints_2d": flat}]}
    out_path = DST / jf.name
    with open(out_path, "w") as f:
        json.dump(out, f)

    # quick sanity check
    if len(flat) != 54:
        print("WARN: not 54 values for", jf.name)
