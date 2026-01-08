import cv2
import csv
from pathlib import Path
from datetime import datetime

# ---------------- Paths ----------------
images_dir = Path("images")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

out_csv = output_dir / "results.csv"

# ---------------- CSV setup ----------------
file_exists = out_csv.exists()
rows = []

# ---------------- Color → Bin mapping ----------------
BIN_MAP = {
    "RED": "BIN A",
    "BLUE": "BIN B",
    "GREEN": "BIN C",
    "UNKNOWN": "BIN C"
}

# ---------------- Helper: classify color ----------------
def classify_color_by_mean_hsv(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.mean(hsv)[:3]

    if h < 10 or h > 160:
        return "RED"
    elif 90 < h < 130:
        return "BLUE"
    elif 35 < h < 85:
        return "GREEN"
    else:
        return "UNKNOWN"

# ---------------- Load all images ----------------
image_files = sorted(images_dir.glob("*.jpg"))
print(f"Found {len(image_files)} images")

obj_id = 1

for img_path in image_files:
    print(f"\nProcessing: {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print("Image not readable")
        continue

    # Preprocess
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Simple blue mask (example object)
    lower = (90, 50, 50)
    upper = (140, 255, 255)
    mask = cv2.inRange(blur, lower, upper)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    out = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = img[y:y+h, x:x+w]

        color = classify_color_by_mean_hsv(roi)
        bin_name = BIN_MAP[color]

        timestamp = datetime.now().isoformat(timespec="seconds")
        rows.append([timestamp, obj_id, color, bin_name, int(area)])

        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"Obj {obj_id}: {color} -> {bin_name}"
        cv2.putText(out, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        obj_id += 1

    cv2.imwrite(str(Path("outputs") / f"{img_path.stem}_annotated.jpg"), out)
    cv2.imshow("Detected + Classified", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------- Write CSV ----------------
with open(out_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["timestamp", "object_id", "color", "bin", "area"])
    writer.writerows(rows)

print(f"\nSaved {len(rows)} objects to {out_csv}")

# ---------------- Summary ----------------
bin_counts = {"BIN A": 0, "BIN B": 0, "BIN C": 0}
for r in rows:
    bin_counts[r[3]] += 1

print("\n--- SUMMARY ---")
for b, c in bin_counts.items():
    print(f"{b}: {c} objects")
