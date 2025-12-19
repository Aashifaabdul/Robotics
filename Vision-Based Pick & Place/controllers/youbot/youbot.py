# ============================================================================
#   RATIONALE
# ----------------------------------------------------------------------------
#  Scenario:
#   - A mobile manipulator (YouBot) with cameras and a gripper clean a
#     table by picking objects and dropping them into a bin at a fixed pose.
#
#  Design overview:
#   • Navigation:
#       - Uses GPS (position) + compass (heading) to drive the base between
#         TABLE_POS, WAYPOINT1 and BIN_STOP_POS.
#       - Navigation is structured in two steps:
#           1) turn_to_target(): rotate in place to face the goal.
#           2) move_straight_to_target(): drive forward with heading correction.
#       - align_base_to_table() stores a "table-facing" heading so every
#         approach to the table has consistent orientation for perception.
#
#   • Object detection & localisation:
#       - _detect_cubes_recognition() uses the Webots camera recognition
#         system to obtain 2D bounding boxes + approximate depth for objects.
#       - capture_depth_frame(), get_camera_intrinsics() and
#         pixel_to_base() use a pinhole camera model + depth image
#         to back-project image pixels (u, v) into 3D points in the robot
#         base frame.
#       - detect_closest_cube() fuses recognition + depth to find the closest cube
# 
#
#   • Manipulation (pick & place):
#       - Arm motion is done in joint space (move_PTP) for safe configurations
#         (HOME_POSE, CARRY_POSE, PRE_PLACE_BIN, PLACE_BIN) and in task space
#         (move_LIN) for precise vertical lifting after grasp.
#       - The pick behaviour is split into two visual-servo phases:
#           PHASE 1 (search_detect_the_red_center_):
#               Rotate base to align the object horizontally with the TCP
#               crosshair (image centre), only controlling yaw.
#           PHASE 2 (visual_servo_grab_with_ideology):
#               Move forward slowly while:
#                 * keeping the object horizontally centered (pixel error),
#                 * doing small lateral corrections using 3D Y offset,
#                 * triggering the grasp when the object is either low in
#                   the image (close to gripper) or near in 3D distance.
#       - close_gripper_to_lift() closes the gripper, lifts in Cartesian
#         space, and moves to CARRY_POSE for safe navigation.
#       - place_in_bin() uses fixed joint poses to open the gripper above
#         the bin and then return to CARRY_POSE.
#
#  This structure clearly separates navigation, perception, and manipulation
#  while achieving autonomous pick-and-place until the table region is empty.
# ============================================================================
import math
import time
import numpy as np
import cv2
import kinpy as kp
from scipy.spatial.transform import Rotation as R

from youbot_base import YouBot
# ====== WORLD / BASE POSES ======
# Predefined waypoints for table and bin in world coordinates.
TABLE_POS = np.array([-1.472, -0.2], dtype=float)
WAYPOINT1 = np.array([-1.472, 0.2], dtype=float)
INITIAL_HEADING_REF = 1.04738

# ====== ARM CONFIGS ======
# Joint-space postures chosen to be collision-safe and repeatable.
HOME_POSE     = np.array([0.0, 2.0, 0.3, -0.85, 0.0], dtype=float) #safe pose above the table for searching/approaching
CARRY_POSE    = np.array([0.0, 0.3, 0.0, 1.5, 0.0], dtype=float)#safe “transport” pose with object in gripper.
PRE_PLACE_BIN = np.array([0.0, 0.3, 0.0, 1.5, 0.0], dtype=float) #intermediate pose before dropping into bin.
PLACE_BIN     = np.array([-0.10, 1.05, 0.65, 1.05, 0.0], dtype=float) #pose where TCP is above bin, good for dropping.

# Bin world pose (same as your earlier mission)
# Bin position is assumed fixed, so navigation to it can be purely GPS-based.
BIN_X = 0.05
BIN_Y = 1.40
BIN_Z = 0.40
BIN_APPROACH_OFFSET = 0.1 #stop 0.1 m before bin 
BIN_STOP_POS = np.array([BIN_X - BIN_APPROACH_OFFSET, BIN_Y], dtype=float)

# ====== VISION PARAMS ======
# Camera resolution and FPS bookkeeping.
IMG_H, IMG_W = 120, 160 #Default image size
_last_fps_t = time.time() #last time we measured FPS.
_last_fps = 0.0 #current FPS estimate

# Global dict for the latest detected cube (for quick access).
LATEST_CUBE = {"bbox": None, "center": None, "score": 0.0, "pos_cam": None, "t": 0.0} #bb- 'xywh', centre - 'uv', score - detection score, pos_cam- 3D positiion in cam frame , t- 'timestep' 
TABLE_HEADING = None  # stored once so table approaches are consistent

def _publish_latest_cube(bbox, center, score, pos_cam=None):
    # Store latest detection with timestamp so others can query "fresh" data.
    LATEST_CUBE.update({"bbox": bbox, "center": center, "score": float(score),
        "pos_cam": None if pos_cam is None else np.array(pos_cam, float), "t": time.time()})

def _clear_latest_cube():
    # Called when no object is visible: prevents acting on stale detections.
    LATEST_CUBE.update({"bbox": None, "center": None, "score": 0.0, "pos_cam": None, "t": time.time()})

def get_latest_cube(max_age_s=0.5):
    # Returns cached cube only if it's younger than max_age_s seconds.
    # If last detection is older than max_age_s, we treat it as invalid.
    now = time.time()
    if LATEST_CUBE["center"] is None or now - LATEST_CUBE["t"] > max_age_s:
        return None, None, 0.0
    return LATEST_CUBE["center"], LATEST_CUBE["bbox"], LATEST_CUBE["score"]

def step_robot(robot, steps=1):
    # Convenience wrapper: advances simulation a fixed number of basic time steps.
    for _ in range(steps):
        if robot.step(robot.TIME_STEP) == -1:
            return False
    return True

def _get_bgr_frame(robot):
    # Grabs raw RGBA data from Webots and converts to BGR image for OpenCV.
    bytess = robot.rgb_camera.getImage()
    if not bytess:
        return None
    rgba = np.frombuffer(bytess, np.uint8).reshape((IMG_H, IMG_W, 4))
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

def _detect_cubes_recognition(robot):
    """
    Object detection (2D) using Webots recognition:
    - Reads recognised objects from the camera.
    - Converts them into bounding boxes and centers in image space.
    - Assigns a simple score based on depth (closer objects are preferred).
    """
    """(Recognition node)
    Each “Object” contains:
        its model name
        its position in the image (u, v)
        its bounding box size (width, height)
        its 3D position relative to the camera
    """
    detections = []
    objs = robot.rgb_camera.getRecognitionObjects()
    if not objs:
        return detections
    for obj in objs:
        u, v = obj.getPositionOnImage() #in pixels
        w_img, h_img = obj.getSizeOnImage() #in pixels
        w = int(max(1, round(w_img))) #bb must be integers 
        h = int(max(1, round(h_img)))
        x = max(0, min(IMG_W - 1, int(round(u - w / 2.0)))) #left edge
        y = max(0, min(IMG_H - 1, int(round(v - h / 2.0)))) #top edge
        if x + w > IMG_W: w = IMG_W - x
        if y + h > IMG_H: h = IMG_H - y
        pos_cam = obj.getPosition() #return xyz position 
        z = float(pos_cam[2]) #forward distance
        # depth-based score: nearer objects get higher score
        score = 1.0 / (1.0 + max(z, 1e-3))
        detections.append({"bbox": (x, y, w, h), "center": (int(round(u)), int(round(v))),
            "score": float(score), "pos_cam": pos_cam})
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections

def _draw_detections(bgr, detections, fps=0.0):
    """
    Visualisation for debugging:
    - Draw bounding boxes and RED DOT at object center.
    - Draw a crosshair at image centre (TCP direction).
    - Overlay FPS and number of detections.
    """
    out = bgr.copy()
    for i, d in enumerate(detections):
        x, y, w, h = d["bbox"]
        cx, cy = d["center"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)  # RED DOT
        cv2.putText(out, f"Cube {i+1} | s={d['score']:.2f}", (x, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    # Crosshair = desired TCP projection in image
    cv2.line(out, (IMG_W // 2, 0), (IMG_W // 2, IMG_H), (255, 255, 255), 1) #centre vertical
    cv2.line(out, (0, IMG_H // 2), (IMG_W, IMG_H // 2), (255, 255, 255), 1) #centre horizontal
    cv2.circle(out, (IMG_W // 2, IMG_H // 2), 5, (0, 255, 255), 2)  # TCP CENTER
    cv2.putText(out, f"Detections: {len(detections)}  FPS: {fps:.1f}", (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def process_camera(robot, detect=True):
    """
    Main camera processing step:
    - Optionally runs recognition-based detection.
    - Maintains a global dict of the best cube.
    - Displays a debug window with overlays and FPS.
    This is called inside all control loops so perception stays up to date.
    """
    global _last_fps_t, _last_fps
    bgr = _get_bgr_frame(robot) #rgba-> bgr
    if bgr is None:
        if detect: _clear_latest_cube()#so we don’t act on old detections.
        return
    if detect:
        detections = _detect_cubes_recognition(robot)
        if detections:
            d0 = detections[0]
            _publish_latest_cube(d0["bbox"], d0["center"], d0["score"], pos_cam=d0["pos_cam"])
        else:
            _clear_latest_cube()
    else:
        detections = []
    now = time.time()
    dt = now - _last_fps_t
    if dt > 0: _last_fps = 1.0 / dt
    _last_fps_t = now
    vis = _draw_detections(bgr, detections, fps=_last_fps) if detect else bgr.copy()
    cv2.imshow("RGB Camera", vis)
    cv2.waitKey(1)

def get_xy(robot):
    # Read robot (x, y) from GPS in world coordinates.
    v = robot.gps.getValues() #x-left/right y - forward/backward
    return np.array([v[0], v[1]], dtype=float)

def get_heading(robot, offset=0.0):
    # Compute yaw from compass, including a calibration offset.
    c = robot.compass.getValues() #cx,cy 
    return (math.atan2(c[0], c[1]) + offset + math.pi) % (2 * math.pi) - math.pi

def normalize_angle(angle): 
    # Wrap any angle to the range [-pi, pi] for stable control.
    return (angle + math.pi) % (2 * math.pi) - math.pi 

def turn_to_target(robot, target_xy, heading_offset):
    """
    Navigation step 1: rotate robot to face the goal position.
    Rationale: separating rotation and translation simplifies control and
    avoids curved or unstable trajectories.
    """
    ROT_KP, ROT_TOL = 2.0, 0.01 # ROT_KP- how aggressively it turns to fix the angle error, ROT_TOL- how accurate the final heading must be
    print(f"  Turning to face ({target_xy[0]:.3f}, {target_xy[1]:.3f})...")
    while robot.step(robot.TIME_STEP) != -1:
        process_camera(robot, detect=True)
        pos = get_xy(robot)
        desired = math.atan2(target_xy[1] - pos[1], target_xy[0] - pos[0])
        error = normalize_angle(desired - get_heading(robot, heading_offset))
        if abs(error) < ROT_TOL:
            robot.set_mecanuum_control(0, 0, 0)
            print(f"   Aligned (error={abs(error):.4f} rad)")
            return True
        robot.set_mecanuum_control(0, 0, float(np.clip(ROT_KP * error, -1.5, 1.5)))
    return False

def move_straight_to_target(robot, target_xy, heading_offset):
    """
    Navigation step 2: drive straight towards the goal with heading correction.
    Rationale: forward velocity is proportional to distance; yaw correction
    term keeps the robot pointing to the target during the approach.
    """
    # Linear controller gain and distance tolerance (m) for position convergence
    KLIN, DIST_TOL = 2.0, 0.005 # KLIN- How fast the robot drives depending on distance , DIST_TOL-If robot is within 0.5 cm of goal → stop.
    print(f"  Moving to ({target_xy[0]:.3f}, {target_xy[1]:.3f})...")
    while robot.step(robot.TIME_STEP) != -1:
        process_camera(robot, detect=True)
        pos = get_xy(robot)
        dist = np.linalg.norm(target_xy - pos) #how far from target
        if dist < DIST_TOL:
            robot.set_mecanuum_control(0, 0, 0)
            print(f"   Position reached (error={dist:.4f} m)")
            return True
        lin_speed = float(np.clip(KLIN * dist, -0.8, 0.8)) 
        target_heading = math.atan2(target_xy[1] - pos[1], target_xy[0] - pos[0]) #atan2(dy,dx)
        angle_err = normalize_angle(target_heading - get_heading(robot, heading_offset))
        robot.set_mecanuum_control(lin_speed, 0.0, angle_err * 2.0)
    return False

def turn_to_heading(robot, target_heading, heading_offset):
    """
    Rotate robot to a specific absolute heading.
    Used to enforce a consistent orientation in front of the table so vision
    and grasping conditions are repeatable across cycles.
    """
    # Rotation control tuning: ROT_KP sets the proportional gain; ROT_TOL sets the acceptable rotational error (radians) before considering the heading on target.
    ROT_KP, ROT_TOL = 1.0, 0.01
    print(f"  Turning to heading {math.degrees(target_heading):.1f}°...")
    while robot.step(robot.TIME_STEP) != -1:
        process_camera(robot, detect=True)
        err = normalize_angle(target_heading - get_heading(robot, heading_offset))
        if abs(err) < ROT_TOL:
            robot.set_mecanuum_control(0.0, 0.0, 0.0)
            step_robot(robot, 5)
            print(f"   Heading reached (err={abs(err):.4f} rad)")
            return True
        robot.set_mecanuum_control(0.0, 0.0, float(np.clip(ROT_KP * err, -0.8, 0.8)))
    return False

def align_base_to_table(robot, heading_offset, tilt_deg=30.0):
    """
    Capture and reuse a "table-facing" heading (with small tilt):
    - On the first call, store the current heading + tilt as TABLE_HEADING.
    - Later calls turn the robot back to this saved orientation.
    This keeps the camera viewpoint on the table nearly constant.
    """
    global TABLE_HEADING
    if TABLE_HEADING is None:
        cur = get_heading(robot, heading_offset)
        TABLE_HEADING = normalize_angle(cur + math.radians(tilt_deg))
        print(f"[TABLE_HEADING] Captured: {math.degrees(TABLE_HEADING):.1f}°")
    print("=== ALIGNING BASE TO STORED TABLE HEADING ===")
    return turn_to_heading(robot, TABLE_HEADING, heading_offset)

def navigate_to_table(robot, heading_offset):
    """
    High-level navigation: move the base to the central table position.
    Sequence:
      1) turn_to_target(TABLE_POS)
      2) move_straight_to_target(TABLE_POS)
      3) align_base_to_table() for consistent view.
    """
    print("=== NAVIGATING TO TABLE CENTER ===")
    if not turn_to_target(robot, TABLE_POS, heading_offset): return False
    if not move_straight_to_target(robot, TABLE_POS, heading_offset): return False
    if not align_base_to_table(robot, heading_offset, tilt_deg=30.0): return False
    robot.set_mecanuum_control(0.0, 0.0, 0.0)
    step_robot(robot, 5)
    print(" At table center\n")
    return True

def navigate_to_waypoint(robot, waypoint, heading_offset):
    """
    Same pattern as navigate_to_table but for an offset waypoint on the table.
    This lets the robot cover different regions of the table surface.
    """
    print(f"=== NAVIGATING TO WAYPOINT ({waypoint[0]:.3f}, {waypoint[1]:.3f}) ===")
    if not turn_to_target(robot, waypoint, heading_offset): return False
    if not move_straight_to_target(robot, waypoint, heading_offset): return False
    if not align_base_to_table(robot, heading_offset, tilt_deg=30.0): return False
    robot.set_mecanuum_control(0.0, 0.0, 0.0)
    step_robot(robot, 5)
    print(" At waypoint\n")
    return True

def navigate_to_bin(robot, heading_offset):
    """
    Navigate to the fixed bin position using the same two-step navigation.
    Since the bin is static, there is no need for vision-based localisation.
    """
    print("=== NAVIGATING TO BIN ===")
    if not turn_to_target(robot, BIN_STOP_POS, heading_offset): return False
    if not move_straight_to_target(robot, BIN_STOP_POS, heading_offset): return False
    robot.set_mecanuum_control(0.0, 0.0, 0.0)
    print(" At bin\n")
    return True

def get_camera_intrinsics():
    """
    convert image pixels to 3D space points
    Build approximate camera intrinsics matrix K from a known FOV.
    Used to back-project image pixels to 3D rays before applying depth.
    """
    fov = 1.0472 # 60 degrees in radians
    f = IMG_W / (2.0 * math.tan(fov / 2.0))
    return np.array([[f, 0.0, IMG_W/2.0], [0.0, f, IMG_H/2.0], [0.0, 0.0, 1.0]])

def convert_cam_to_base_frame(P_c):
    """
    The camera coordinate frame and robot base coordinate frame use different axes.
    Convert point from camera frame to robot base frame.
    This accounts for different axis conventions (camera vs base).
    """
    return np.array([P_c[2], -P_c[0], -P_c[1]])

def capture_depth_frame(robot):
    # Fetch depth image and reshape into (H, W) array.
    w = robot.depth_camera.getWidth()
    h = robot.depth_camera.getHeight()
    buf = robot.depth_camera.getRangeImage()
    return np.array(buf, dtype=float).reshape((h, w))

def pixel_to_base(robot, u, v, depth_image, K):
    """
    Pixel + depth -> 3D point in base frame.
    Steps:
      1) Read depth at pixel (u, v), fall back to local median if invalid.
      2) Use K to compute (x, y, z) in camera coordinates.
      3) Transform from camera frame to base frame via forward kinematics.
    """
    h, w = depth_image.shape
    u = int(np.clip(u, 0, w - 1))#If detection gave a pixel slightly outside the image, we clip it.
    v = int(np.clip(v, 0, h - 1))
    d = float(depth_image[v, u]) #distance from camera → object.
    if d <= 0.0 or math.isinf(d) or math.isnan(d):
        window = 3
        patch = depth_image[max(0,v-window):min(h,v+window+1), max(0,u-window):min(w,u+window+1)] #Depth cameras sometimes have bad readings. We fix by taking the median of neighbors.
        valid = patch[(patch > 0.1) & (patch < 3.0) & np.isfinite(patch)]
        d = float(np.median(valid)) if valid.size > 0 else 0.5
    x_cv = (u - K[0,2]) * d / K[0,0] #we know which pixel , how far -> 3D position 
    y_cv = (v - K[1,2]) * d / K[1,1]
    P_c_robot = convert_cam_to_base_frame(np.array([x_cv, y_cv, d]))
    T_base_cam = robot.forward_kinematics(robot.GRIPPER_ARM, camera_link=True)
    T_cam_point = kp.Transform(pos=P_c_robot, rot=[1.0, 0.0, 0.0, 0.0])
    return np.array((T_base_cam * T_cam_point).pos)

def get_tcp_base_xyz(robot):
    # Returns TCP position in base frame from forward kinematics.
    fk = robot.forward_kinematics(robot.GRIPPER_ARM, joint_pos=robot.joint_pos(robot.GRIPPER_ARM))
    return np.array(fk.pos, float) #fk.pos- actual 3D co-ordinates of gripper 

def halt_base_motion(robot):
    # Emergency helper: stop all base motion immediately.
    robot.set_mecanuum_control(0.0, 0.0, 0.0)

def move_arm_to_pregrasp(robot):
    """
    Move arm into a safe HOME_POSE and open the gripper.
    This gives a consistent starting posture for each picking attempt.
    """
    robot.move_PTP(robot.GRIPPER_ARM, HOME_POSE, timeout=2.0, velocity=0.7)
    step_robot(robot, 5)
    robot.open_gripper()
    step_robot(robot, 5)

def close_gripper_to_lift(robot, lift_height=0.07):
    """
    Close gripper and lift vertically in task space before going to CARRY_POSE.
    Lifting in Cartesian space reduces risk of dragging the cube along the table.
    """
    robot.close_gripper()
    step_robot(robot, 30)
    cur = robot.forward_kinematics(robot.GRIPPER_ARM)
    lift_tf = kp.Transform(pos=cur.pos + np.array([0.0, 0.0, lift_height]), rot=cur.rot)
    robot.move_LIN(robot.GRIPPER_ARM, lift_tf, velocity=0.05)
    step_robot(robot, 20)
    robot.move_PTP(robot.GRIPPER_ARM, CARRY_POSE, timeout=2.0, velocity=0.6)
    step_robot(robot, 10)

def detect_closest_cube(robot, K, max_dist=1.8):
    """
    Fuse recognition + depth:
    - Filter out known non-target models (bin, table, floor, ground).
    - Project each remaining object's image position into base frame.
    - Return the closest one in the XY plane, capped by max_dist.
    This selects the next object on the table to clean.
    """
    depth = capture_depth_frame(robot)
    objs = robot.rgb_camera.getRecognitionObjects()
    if not objs:
        return None
    cand = []
    for obj in objs:
        name = obj.getModel().lower()
        if any(x in name for x in ["bin", "trash", "table", "floor", "ground"]):
            continue
        u, v = obj.getPositionOnImage()
        pos_b = pixel_to_base(robot, u, v, depth, K)
        dist_xy = float(np.linalg.norm(pos_b[:2]))
        if dist_xy <= max_dist:
            cand.append({"model": obj.getModel(), "u": float(u), "v": float(v), "pos": pos_b, "dist": dist_xy})
    if not cand:
        return None
    cand.sort(key=lambda c: c["dist"])
    return cand[0]

def search_detect_the_red_center_(robot, K, pixel_tol=8.0, max_steps=800):
    """PHASE 1: Rotate until RED DOT aligns with TCP crosshair (image centre)."""
    width = robot.rgb_camera.getWidth()
    cx = width / 2.0
    steps = 0
    print("\n[PHASE 1] Aligning RED DOT with TCP ...")
    while robot.step(robot.TIME_STEP) != -1 and steps < max_steps:
        process_camera(robot, detect=True)
        t = detect_closest_cube(robot, K, max_dist=3.0)
        if t is None:
            # If nothing is visible, slowly spin to search the table.
            robot.set_mecanuum_control(0.0, 0.0, 0.25)
            steps += 1
            continue
        err = t["u"] - cx #how far left/right the cube is from the centre
        if abs(err) < pixel_tol:
            robot.set_mecanuum_control(0.0, 0.0, 0.0)
            print(f"  RED DOT aligned! (err={err:.1f}px)\n")
            return t
        # Still not aligned → we rotate in place.
        robot.set_mecanuum_control(0.0, 0.0, float(np.clip(-0.006 * err, -0.6, 0.6)))
        steps += 1
    robot.set_mecanuum_control(0.0, 0.0, 0.0)
    return None

def visual_servo_grab_with_ideology(robot, K, max_steps=870):
    """
    PHASE 2: With RED DOT centred, approach and grasp with continuous servoing.
    
    Strategy:
    - Use image-space error (u - cx) to control yaw (rotation around z).
    - Use 3D Y position (side offset) to apply small lateral velocity.
    - Move forward at a slow, constant speed towards the cube.
    - Trigger the grasp only when:
        * the object is horizontally centred (within FINGER_CENTER_TOL), AND
        * it is either low in the image (large v) OR within a close distance.
    - Before closing, perform a short "creep" to let the object sit deeper
      between the fingers for a more reliable grasp.
    """
    width = robot.rgb_camera.getWidth()
    height = robot.rgb_camera.getHeight()
    cx = width / 2.0 #center x-pixel of image
    
    # Thresholds ( ideology adapted)
    GRIP_ROW_THRESHOLD = 92  # how low in the image the cube should be
    FINGER_CENTER_TOL = 10.0  # horizontal centering , how centered (in pixels) it must be
    GRIP_DIST_THRESHOLD = 0.45  # OR cube is close in XY plane (meters)
    MIN_STEPS = 80  # safety: don't grip too early
    
    print("\n[PHASE 2] Approaching cube with continuous centering...")
    print(f"  Grip triggers: v>={GRIP_ROW_THRESHOLD}px OR dist<={GRIP_DIST_THRESHOLD}m")
    print(f"  Must be centered (err_u<={FINGER_CENTER_TOL}px)\n")
    
    step_counter = 0
    while robot.step(robot.TIME_STEP) != -1 and step_counter < max_steps:
        process_camera(robot, detect=True)
        t = detect_closest_cube(robot, K, max_dist=2.0)
        
        if t is None:
            print("  [APPROACH] Lost cube!")
            robot.set_mecanuum_control(0.0, 0.0, 0.0)
            return False
        
        u = t["u"] #x-pixel
        v = t["v"] #y-pixel
        pos_b = t["pos"] #[x,y,z] in base frame 
         
        # Horizontal pixel error (keep cube centered)
        err_u = u - cx
        
        # Distance in XY plane (base→cube)
        dist_xy = float(np.linalg.norm(pos_b[:2]))
        
        # Lateral offset in base frame (for small corrections)
        side = float(pos_b[1])
        
        # Control commands () 
        # 1. Yaw: keep cube at image center
        w = float(np.clip(-0.005 * err_u, -0.4, 0.4))
        
        # 2. Lateral: small correction based on Y in base frame
        vy = float(np.clip(-0.6 * side, -0.10, 0.10))
        
        # 3. Forward: slow constant speed
        vx = 0.035
        
        # Check grip conditions (ideology: row OR distance)
        centered_ok = (abs(err_u) <= FINGER_CENTER_TOL)
        row_ok = (v >= GRIP_ROW_THRESHOLD)
        prox_ok = (dist_xy <= GRIP_DIST_THRESHOLD)
        
        print(f"  [APPROACH] step={step_counter} | u={u:.1f} v={v:.1f} err_u={err_u:.1f} | "
              f"dist_xy={dist_xy:.3f}  | "
              f"centered={centered_ok}")
        
        # GRIP TRIGGER (logic: centered AND (row OR proximity))
        if (step_counter >= MIN_STEPS) and centered_ok and (row_ok or prox_ok):
            print(f"\n  GRIP CONDITIONS MET!")
            print(f"    - Centered: {centered_ok}")
            print(f"    - Row OK: {row_ok} (v={v:.1f}>={GRIP_ROW_THRESHOLD})")
            print(f"    - Proximity OK: {prox_ok} (dist={dist_xy:.3f}<={GRIP_DIST_THRESHOLD})")
            
            # Stop
            robot.set_mecanuum_control(0.0, 0.0, 0.0)
            step_robot(robot, 2)
            
            # Short creep (ideology: get cube deeper between fingers)
            print(" Final approach...")
            creep_steps = 80
            for creep_i in range(creep_steps):
                if robot.step(robot.TIME_STEP) == -1:
                    break
                process_camera(robot, detect=True)
                t2 = detect_closest_cube(robot, K, max_dist=2.0)
                if t2 is None:
                    break
                err2 = t2["u"] - cx
                side2 = float(t2["pos"][1])
                w2 = float(np.clip(-0.005 * err2, -0.3, 0.3))
                vy2 = float(np.clip(-0.6 * side2, -0.07, 0.07))
                vx2 = 0.03  # gentle forward during creep
                robot.set_mecanuum_control(vx2, vy2, w2)
            
            # Close and lift
            robot.set_mecanuum_control(0.0, 0.0, 0.0)
            step_robot(robot, 2)
            print("  [GRIP] Closing gripper...")
            close_gripper_to_lift(robot, lift_height=0.07)
            return True
        
        # Continue approach
        robot.set_mecanuum_control(vx, vy, w)
        step_counter += 1
    
    robot.set_mecanuum_control(0.0, 0.0, 0.0)
    print("  [APPROACH] Max steps reached")
    return False

def pick_with_improved_method(robot):
    """
    High-level pick pipeline (autonomous manipulator behaviour):
    1. Move arm to HOME_POSE and open gripper (move_arm_to_pregrasp).
    2. PHASE 1: search_detect_the_red_center_ to align cube horizontally.
    3. PHASE 2: visual_servo_grab_with_ideology to approach, creep and grasp.
    """
    K = get_camera_intrinsics()
    move_arm_to_pregrasp(robot)
    
    # PHASE 1: Center
    centered = search_detect_the_red_center_(robot, K, pixel_tol=8.0, max_steps=800)
    if centered is None:
        print("[PICK] No centered cube")
        return False
    
    # PHASE 2: Approach and grasp 
    success = visual_servo_grab_with_ideology(robot, K, max_steps=870)
    
    if success:
        print("\n PICK SUCCESSFUL!\n")
    else:
        print("\n PICK FAILED\n")
    
    return success

def place_in_bin(robot):
    """
    Fixed place routine:
    - Move arm to PRE_PLACE_BIN, then PLACE_BIN above the bin, open gripper,
      and finally return to CARRY_POSE.
    Since the bin is static and the base is at BIN_STOP_POS, joint-space
    poses are sufficient to reliably drop the cube inside.
    """
    print("\n=== PLACING CUBE IN BIN ===")
    robot.move_PTP(robot.GRIPPER_ARM, PRE_PLACE_BIN, timeout=3.0, velocity=0.5)
    step_robot(robot, 30)
    robot.move_PTP(robot.GRIPPER_ARM, PLACE_BIN, timeout=3.0, velocity=0.3)
    step_robot(robot, 40)
    robot.open_gripper()
    step_robot(robot, 40)
    robot.move_PTP(robot.GRIPPER_ARM, CARRY_POSE, timeout=2.0, velocity=0.5)
    step_robot(robot, 20)
    print(" Cube placed in bin\n")
    return True

def main():
    """
    Main autonomous cleaning loop:
    - Initialise robot, camera window, compass offset and arm posture.
    - PHASE 1: work from TABLE_POS for up to MAX_TABLE_CYCLES objects.
    - PHASE 2: move to WAYPOINT1 and repeat for MAX_WAYPOINT_CYCLES.
    Each cycle:
        1) Navigate to table/waypoint (navigation module).
        2) Run pick_with_improved_method (vision + manipulation).
        3) Navigate to bin and call place_in_bin (navigation + manipulation).
    The loop terminates when cycles are exhausted or picking fails
    (no more visible cubes), effectively cleaning the table region.
    """
    global IMG_W, IMG_H
    robot = YouBot()
    cv2.namedWindow("RGB Camera", cv2.WINDOW_NORMAL)
    IMG_W = robot.rgb_camera.getWidth()
    IMG_H = robot.rgb_camera.getHeight()
    step_robot(robot, 2)
    heading_offset = INITIAL_HEADING_REF - get_heading(robot, 0.0)
    print("=" * 60)
    print("YOUBOT: PICK AND PLACE STARTED")
    print("=" * 60 + "\n")
    print("INITIALIZING ARM")
    robot.move_PTP(robot.GRIPPER_ARM, CARRY_POSE, timeout=3.0, velocity=0.6)
    step_robot(robot, 12)
    print(" Arm ready in CARRY_POSE\n")
    
    try:
        total_cycles = 0
        table_cycles = 0
        waypoint_cycles = 0
        MAX_TABLE_CYCLES = 3
        MAX_WAYPOINT_CYCLES = 4
        
        # Phase 1: Work at TABLE_POS for 3 cycles
        print("\n" + "=" * 60)
        print("PHASE 1: Working at TABLE_POS")
        
        while table_cycles < MAX_TABLE_CYCLES:
            total_cycles += 1
            table_cycles += 1
            print("\n" + "=" * 60)
            print(f"CYCLE {total_cycles} (TABLE {table_cycles}/{MAX_TABLE_CYCLES})")
            print("=" * 60 + "\n")
            
            # Navigate to table (navigation module)
            if not navigate_to_table(robot, heading_offset):
                print(" Could not reach table")
                break
            
            # Pick cube from table (perception + manipulation)
            ok_pick = pick_with_improved_method(robot)
            if not ok_pick:
                print(f" Pick failed at TABLE_POS")
                # Move arm to CARRY_POSE
                print(" Moving arm to CARRY_POSE...")
                robot.move_PTP(robot.GRIPPER_ARM, CARRY_POSE, timeout=2.0, velocity=0.6)
                step_robot(robot, 10)
                print(" Skipping to WAYPOINT1 phase early...")
                break
            
            # Navigate to bin
            print("\n>>> Moving to bin with cube in CARRY_POSE...")
            if not navigate_to_bin(robot, heading_offset):
                print(" Could not reach bin")
                break
            
            # Place cube in bin
            place_in_bin(robot)
            print(f"\n CYCLE {total_cycles} COMPLETE!\n")
            step_robot(robot, 20)
        
        # Phase 2: Work at WAYPOINT1 for 4 cycles
        print("\n" + "=" * 60)
        print("PHASE 2: Switching to WAYPOINT1")
        print("=" * 60 + "\n")
        
        while waypoint_cycles < MAX_WAYPOINT_CYCLES:
            total_cycles += 1
            waypoint_cycles += 1
            print("\n" + "=" * 60)
            print(f"CYCLE {total_cycles} (WAYPOINT {waypoint_cycles}/{MAX_WAYPOINT_CYCLES})")
            print("=" * 60 + "\n")
            
            # Navigate to waypoint
            if not navigate_to_waypoint(robot, WAYPOINT1, heading_offset):
                print(" Could not reach waypoint")
                break
            
            # Pick cube from waypoint
            ok_pick = pick_with_improved_method(robot)
            if not ok_pick:
                print(f" Pick failed at WAYPOINT1")
                # Move arm to CARRY_POSE
                print(" Moving arm to CARRY_POSE...")
                robot.move_PTP(robot.GRIPPER_ARM, CARRY_POSE, timeout=2.0, velocity=0.6)
                step_robot(robot, 10)
                print(" No more cubes at WAYPOINT1 - ending")
                break
            
            # Navigate to bin
            print("\n Moving to bin with cube in CARRY_POSE...")
            if not navigate_to_bin(robot, heading_offset):
                print(" Could not reach bin")
                break
            
            # Place cube in bin
            place_in_bin(robot)
            print(f"\n CYCLE {total_cycles} COMPLETE!\n")
            step_robot(robot, 20)
        
        print("\n" + "=" * 60)
        print("ALL CYCLES COMPLETE!")
        print(f"Total cycles: {total_cycles}")
        print(f"- TABLE_POS: {table_cycles} cycles")
        print(f"- WAYPOINT1: {waypoint_cycles} cycles")
        print("=" * 60 + "\n")
        
        print("\n=== MISSION COMPLETE ===")
        # Keep camera running briefly so final state can be inspected visually.
        for _ in range(50):
            if robot.step(robot.TIME_STEP) == -1: break
            process_camera(robot, detect=True)
            time.sleep(0.05)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
