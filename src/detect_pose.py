import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector
from pathlib import Path
import math

def load_camera_params(yaml_path: str):
    """
    Loads camera intrinsics K and distortion coeffs dist from a YAML file.
    Expected format:
      camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
      dist_coeffs: [k1, k2, p1, p2, k3]  # or length 4/8 depending on calibration
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return K, dist

def euler_from_rotation_matrix(R: np.ndarray):
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees.

    Convention:
      - OpenCV camera frame: +x right, +y down, +z forward
      - roll  about +z
      - pitch about +x
      - yaw   about +y
    This is one common robotics-ish interpretation; the key is consistency.
    """
    # Guard numeric issues
    sy = math.sqrt(R[0,0]*R[0,0] + R[2,0]*R[2,0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2,1], R[2,2])     # about x
        yaw   = math.atan2(-R[2,0], sy)        # about y
        roll  = math.atan2(R[1,0], R[0,0])     # about z
    else:
        pitch = math.atan2(-R[1,2], R[1,1])
        yaw   = math.atan2(-R[2,0], sy)
        roll  = 0.0

    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))

def draw_tag_box(frame, corners, color=(0, 255, 0), thickness=2):
    # corners: (4,2) in order provided by detector
    pts = corners.astype(int).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

def draw_axes(frame, K, dist, rvec, tvec, axis_len=0.05):
    """
    Draw 3D axes on the tag: X=red, Y=green, Z=blue.
    axis_len in meters.
    """
    axis_3d = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ], dtype=np.float64)

    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    origin = tuple(imgpts[0])
    cv2.line(frame, origin, tuple(imgpts[1]), (0, 0, 255), 2)  # X red
    cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 2)  # Y green
    cv2.line(frame, origin, tuple(imgpts[3]), (255, 0, 0), 2)  # Z blue

def main():
    # === CONFIG ===
    tag_family = "tag36h11"
    tag_size_m = 0.10  # <-- SET THIS: physical tag side length in meters
    camera_yaml = str(Path(__file__).parent / "camera.yaml")
    use_calibrated = Path(camera_yaml).exists()

    # Webcam index (0 is typical)
    cam_index = 0

    # AprilTag detector
    detector = Detector(
        families=tag_family,
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try cam_index=1 or check permissions.")

    # If calibrated, load camera intrinsics; else build a rough guess from frame size
    K = None
    dist = None

    print("Press 'q' to quit.")
    print(f"Calibration file found: {use_calibrated} ({camera_yaml})")

    # 3D model points for tag corners in tag frame, centered at origin.
    # Must match the detector corner ordering.
    # pupil-apriltags provides corners in order: [top-left, top-right, bottom-right, bottom-left] (typically)
    # We'll assume that common ordering; if overlay looks mirrored, we’ll swap accordingly.
    s = tag_size_m
    half = s / 2.0
    obj_pts = np.array([
        [-half, -half, 0],  # corner 0
        [ half, -half, 0],  # corner 1
        [ half,  half, 0],  # corner 2
        [-half,  half, 0],  # corner 3
    ], dtype=np.float64)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lazy-init intrinsics if not calibrated
        if K is None:
            h, w = frame.shape[:2]
            if use_calibrated:
                K, dist = load_camera_params(camera_yaml)
            else:
                # Rough guess: fx ~ fy ~ 0.9*w, principal point center, no distortion
                fx = 0.9 * w
                fy = 0.9 * w
                cx = w / 2.0
                cy = h / 2.0
                K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float64)
                dist = np.zeros((5, 1), dtype=np.float64)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray, estimate_tag_pose=False)

        # UI header box
        overlay_text = [
            "AprilTag Pose Detector",
            f"Tag family: {tag_family} | Tag size: {tag_size_m:.3f} m",
            "Calibrated: YES" if use_calibrated else "Calibrated: NO (approx intrinsics)"
        ]

        y0 = 25
        for i, t in enumerate(overlay_text):
            cv2.putText(frame, t, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if len(detections) > 0:
            # For demo: pick the detection with largest pixel area (closest tag)
            def det_area(det):
                c = det.corners
                # polygon area
                return abs(cv2.contourArea(c.astype(np.float32)))
            det = max(detections, key=det_area)

            corners = det.corners.astype(np.float64)  # shape (4,2)
            draw_tag_box(frame, corners)

            # solvePnP wants image points shape (N,1,2)
            img_pts = corners.reshape(-1, 1, 2)

            # Use iterative PnP for stability; could switch to SOLVEPNP_IPPE_SQUARE as well.
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if ok:
                # Translation vector: camera -> tag center in meters (same units as obj_pts)
                tx, ty, tz = tvec.flatten()
                distance = float(np.linalg.norm(tvec))  # range to center

                # Rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                pitch_deg, yaw_deg, roll_deg = euler_from_rotation_matrix(R)

                # Draw axes (optional but great for demo)
                draw_axes(frame, K, dist, rvec, tvec, axis_len=0.05)

                # Draw center point
                center_px = tuple(det.center.astype(int))
                cv2.circle(frame, center_px, 4, (0, 255, 255), -1)

                # Text block
                info = [
                    f"ID: {det.tag_id}",
                    f"Distance to center: {distance:.3f} m (z={tz:.3f} m)",
                    f"Rotation (deg): X(pitch)={pitch_deg:+.1f}  Y(yaw)={yaw_deg:+.1f}  Z(roll)={roll_deg:+.1f}"
                ]
                y1 = 120
                for i, line in enumerate(info):
                    cv2.putText(frame, line, (10, y1 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "solvePnP failed", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No AprilTag detected", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("AprilTag_PoseDetector", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

