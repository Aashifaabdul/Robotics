#   Summit XL – DIJKSTRA NAVIGATION + LOCAL AVOIDANCE SYSTEM
#   Structured into:
#       1. Global Navigation (Mapping + [Dijkstra] Path Planning)
#       2. Local Navigation  (Waypoint Following)
#       3. Obstacle Avoidance (Reactive Layer)
#       4. Behaviour Coordination Strategy (State Machine)

#SYSTEM ARCHITECTURE :
    #Deliberative Layer (Global Planner)
        #Responsible for generating a collision-free path from the robot’s current position to the target goal using an occupancy grid and 
        #A shortest-path algorithm (Dijkstra depending on variation tested).
        #This layer provides structured, optimal navigation over the static layout of the museum.

    #Reactive Layer (Local Obstacle Avoidance)
        #Handles real-time avoidance of dynamic objects including moving obstacles and pedestrians.
        #This layer uses LIDAR scans to detect obstacles, evaluate free-space clearance, perform rotational scanning,
        #and execute mecanum-based lateral dodges or emergency manoeuvres when required.

    #Executive Layer (Behaviour Coordinator)
        #A state machine that switches between navigation, avoidance, scanning, and recovery states.
        #It ensures the system reacts adaptively to interruptions, dynamic obstacles, and new user commands.

import math
import numpy as np
import cv2
import heapq
from controller import Robot
from keyboardreader import KeyboardReader


# ============================
#  WORLD DIMENSIONS (Webots)
# ============================
#Real world map size:
REAL_WIDTH_M  = 12.0   # X-axis (world width in meters (X axis))
REAL_HEIGHT_M = 14.0   # Y-axis (world height in meters (Y axis))

# =============================================================
#   ROBOT SETUP
# =============================================================
WHEEL_RADIUS = 0.123
LX = 0.2045 #how far the wheels are  from the centre of the robot 
LY = 0.2225
MAX_SPEED = 20

# obstacle avoidance parameters 
CRITICAL_DISTANCE = 0.9 # when we begin scanning
EMERGENCY_DISTANCE = 0.5 # sideways emergency dodge
SAFE_DISTANCE = 1.8 # slow down distance
GOAL_REACH_DISTANCE = 1.5
WAYPOINT_REACH_DISTANCE = 0.3

#Dijkstra Parameters 
FREE_THRESH = 0.6
REPLAN_INTERVAL = 80 # plannign interval
INFLATION_RADIUS = 5  # pixels to inflate obstacles
# Scan failure counters
scan_fail_count = 0
SCAN_FAIL_LIMIT = 4 
robot = Robot()
timestep = int(robot.getBasicTimeStep())

#===================================================
#ROBOT'S DEVICES
#===================================================

motors = [
    robot.getDevice("front_left_wheel_joint"),
    robot.getDevice("front_right_wheel_joint"),
    robot.getDevice("back_left_wheel_joint"),
    robot.getDevice("back_right_wheel_joint")
]

for m in motors:
    m.setPosition(math.inf)
    m.setVelocity(0)

#===================================================
# SENSORS
#===================================================
lidar = robot.getDevice("Hokuyo UTM-30LX")
lidar.enable(timestep)

gps = robot.getDevice('gps')
gps.enable(timestep)

compass = robot.getDevice('compass')
compass.enable(timestep)

camera = robot.getDevice("rgb_camera")
camera.enable(timestep)

#CAMERA WINDOW 
cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

keyboard = KeyboardReader(timestep)

#MAPPING AND ENVIRONMENTAL REPRESENTATION

#The environment is represented using a binary occupancy grid derived from a static map.
#This representation was selected because:
    #It is directly compatible with grid-based path planning algorithms taught in the module.
    #It abstracts spatial information cleanly, separating free regions from obstacles.
    #It allows efficient recomputation of global paths when the robot relocates or goals change.

# =============================================================
#   LOAD AND PROCESS MAP
# =============================================================
 # CREATED OCCUPANCY GRID USING THIS CODE 
""" import cv2
        import numpy as np
        img = cv2.imread("map.png", cv2.IMREAD_GRAYSCALE)

        # Convert to occupancy grid (0 = free, 1 = wall)

        _, occ = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV) 
        cv2.imshow("Occupancy Grid", occ * 255)
        np.save("occupancy_grid.npy", occ)
        
"""
try:
    occupancy_map = np.load("occupancy_grid.npy").astype(np.float32)
except:
    occupancy_map = np.ones((650, 654), dtype=np.float32) * 0.5

H, W = occupancy_map.shape 

# Normalizing
if occupancy_map.max() > 1:
    occupancy_map /= 255.0

# Inflate obstacles for safer planning
def inflate_obstacles(occ_map, radius=INFLATION_RADIUS):
    """Inflate obstacles to create safety margin"""
    inflated = occ_map.copy()
    obstacle_mask = (occ_map < FREE_THRESH)
    #this is the inflation neigbourhood
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dx*dx + dy*dy <= radius*radius: #circle inflation for smoother inflation 
                shifted = np.roll(np.roll(obstacle_mask, dy, axis=0), dx, axis=1) #np.roll() moves obstacles by (dx, dy)
                inflated = np.minimum(inflated, np.where(shifted, 0.0, inflated))
    
    return inflated

occupancy_map_inflated = inflate_obstacles(occupancy_map) #Obstacles inflated for safety


"""GPS gives position (meters)
   Dijkstra uses positions (pixels)
    
"""
# Map origin
MAP_ORIGIN_X = -REAL_WIDTH_M / 2.0   # -12/2 = -6 top-left x: from -width/2 to +width/2  , y: from +height/2 downwards
MAP_ORIGIN_Y =  REAL_HEIGHT_M / 2.0   #14/2 = +7:  (-6,+7) = pixeel(0,0)

# Cell size ,if moved by one pixel , how much moved in real world
CELL_SIZE_X = REAL_WIDTH_M  / W #1 pixel = about 1.8 cm of real world. (12/654)
CELL_SIZE_Y = REAL_HEIGHT_M / H  


# =============================================================
#   COORDINATE TRANSFORMS
# =============================================================
# These functions translate between real-world meter coordinates and pixel coordinates of the occupancy grid,
#  so the robot can plan paths on the map and then execute them in the physical world.

def real_to_map(real): #
    """Convert world (meters) → map (pixels)"""
    x, y = real[0], real[1]
    mx = int((x - MAP_ORIGIN_X) / CELL_SIZE_X)
    my = int((MAP_ORIGIN_Y - y) / CELL_SIZE_Y) #Because: On pictures → going down increases Y ; In real life → going up increases Y
    return mx, my

def map_to_real(mx, my):
    """Convert map (pixels) → world (meters)"""
    x = mx * CELL_SIZE_X + MAP_ORIGIN_X
    y = MAP_ORIGIN_Y - my * CELL_SIZE_Y
    return x, y


# =============================================================
#   ROBOT STATE FUNCTIONS
# =============================================================
# get_robot_position() returns the robot’s real-time (x, y, z) location from the GPS, 
# while get_robot_heading() computes the robot’s facing direction by converting the compass vector into an angle. 
# normalize_angle() keeps any angle wrapped cleanly between –π and +π so the controller can compare directions without sudden jumps.

def get_robot_position():
    return gps.getValues()

def get_robot_heading():
    vals = compass.getValues()
    return math.atan2(vals[0], vals[1])

def normalize_angle(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


# =============================================================
#   MECANUM DRIVE
# =============================================================

def mecanum(vx, vy, w):

    """ vx - forward /backward speed 
    vy - sideway (left/right speed)
    w- rotational speed """
    
    motors[0].setVelocity((vx - vy - (LX + LY)*w) / WHEEL_RADIUS)
    motors[1].setVelocity((vx + vy + (LX + LY)*w) / WHEEL_RADIUS)
    motors[2].setVelocity((vx + vy - (LX + LY)*w) / WHEEL_RADIUS)
    motors[3].setVelocity((vx - vy + (LX + LY)*w) / WHEEL_RADIUS)


# =============================================================
#   LIDAR ANALYSIS
# =============================================================
# This function breaks the LiDAR scan into front, left, and right sectors and finds the closest obstacle in each region, 
# filtering out noisy or invalid distances. 
# It then returns the minimum distance in each sector and triggers an emergency flag if something is dangerously close in front.

def analyze_lidar(ranges):
    if not ranges:
        return {"front": 999, "left": 999, "right": 999, "emergency": False} #999- far away 
    
    w = len(ranges)
    mid = w // 2

    # Narrow front sector to avoid robot body / noise
    front = ranges[mid-4:mid+4]
    front = [d for d in front if 0.15 < d < 30] #lesser - sensor noise , larger - meaningless 
    front_min = min(front) if front else 999 #min- take nearest obs 

    # Left sector
    left = ranges[int(0.70*w):int(0.90*w)] #left enough 
    left = [d for d in left if 0.1 < d < 30]
    left_min = min(left) if left else 999
    
    # Right sector
    right = ranges[int(0.10*w):int(0.30*w)] #right enough
    right = [d for d in right if 0.1 < d < 30]
    right_min = min(right) if right else 999

    return {
        "front": front_min,
        "left": left_min,
        "right": right_min,
        "emergency": front_min < EMERGENCY_DISTANCE
    }


# =============================================================
#  GLOBAL NAVIGATION  (Mapping + Path Planning using Dijkstra)
# =============================================================

#GLOBAL PATH PLANNING
#The global planner uses Dijkstra’s algorithm, which was covered in the module as optimal search strategies over discrete maps.

#Why grid-based planning?
    #Planning is deterministic, optimal, and repeatable.
    #Easily integrates with an occupancy grid.
    #Supports replanning when a new goal is added or when dynamic obstacles force detours.
    

#Why not A* for the final solution?
   #A* was evaluated but did not provide a clear performance benefit in this environment.Due to frequent replanning, grid discretization effects, and safety inflation, 
       #the heuristic-driven behaviour often produced less stable waypoint sequences for the local controller.
        #Dijkstra, while more exhaustive, produced more consistent and repeatable paths in narrow indoor corridors,
         #and was therefore selected.


def dijkstra(start, goal):
    # This function performs global path planning by running Dijkstra’s algorithm on the inflated occupancy grid,
    # expanding outward cell-by-cell until it reaches the goal. 
    # It only visits free cells, keeps track of the cheapest cost to each neighbour, 
    # and finally reconstructs the optimal path by backtracking through the came_from links.
    """Dijkstra pathfinding using inflated map"""
    sx, sy = start # map co-ordinates 
    gx, gy = goal

    # Check bounds
    """ W, H are width and height of the occupancy map.
        x is between 0 and W-1
        y is between 0 and H-1"""
    if not (0 <= sx < W and 0 <= sy < H):
        return None
    if not (0 <= gx < W and 0 <= gy < H):
        return None
    
    # Priority queue: (cost, x, y)
    open_list = [(0, sx, sy, None)] #the frontier (start, x, y, parent) # none - no previous cell
    g_cost = {(sx, sy): 0} # Dictionary that stores the best known cost to reach each cell:
    came_from = {} #stores parent:the cell you came from along the best path.

    # 8-connected neighbors with costs # diagonal = sqrt(1² + 1²) = sqrt(2) ≈ 1.414 
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414) 
    ] #(dx, dy, cost)   diagonal steps are longer than the straight steps so sqrt of 2 

    while open_list:
        cost, x, y, parent = heapq.heappop(open_list)
        if (x, y) in came_from:
            continue# if already explored skip that 
        came_from[(x, y)] = parent

        # Goal reached
        if (x, y) == (gx, gy):
            break

        # Explore neighbors
        for dx, dy, move_cost in neighbors:
            nx, ny = x + dx, y + dy #exploring the neighbouring cells 

            # Check bounds
            if not (0 <= nx < W and 0 <= ny < H):
                continue

            # Check if free
            if occupancy_map_inflated[ny, nx] < FREE_THRESH:
                continue

            ng = cost + move_cost #cost so far + cost to move from current cell to neighbour (1.0 or 1.414).

            if (nx, ny) not in g_cost or ng < g_cost[(nx, ny)]:
                g_cost[(nx, ny)] = ng
                heapq.heappush(open_list, (ng, nx, ny, (x, y)))

    # Check if goal was reached
    if (gx, gy) not in came_from:
        print("[DIJKSTRA] No path found!")
        return None

    # if not Reconstruct path 
    path = []
    cur = (gx, gy)
    while cur is not None:
        path.append(cur)
        cur = came_from[cur] # goal -> start 
    
    path.reverse() #start -> goal 
    print(f"[DIJKSTRA] Path found with {len(path)} waypoints")
    return path


# simplify_path function reduces the number of waypoints by keeping only every n-th point, 
# which removes unnecessary zig-zags created by grid-based planning.
# Fewer waypoints make the robot’s motion smoother and easier to follow without constantly turning.
def simplify_path(path, step=10):
    """Reduce waypoints for smoother navigation"""
    if not path or len(path) <= 2:
        return path
    
    simplified = [path[0]]
    for i in range(step, len(path), step):
        simplified.append(path[i])
    
    if simplified[-1] != path[-1]:
        simplified.append(path[-1])
    
    return simplified


# =============================================================
#   LOCAL NAVIGATION (Waypoint Following + Mecanum Control)
# =============================================================
# handles local waypoint tracking by steering the robot toward the next target while dynamically adjusting speed 
# rotation based on direction error and obstacle proximity. 
# It advances to the next waypoint once close enough, slows down near obstacles, 
# and rotates in place if the robot is facing the wrong direction to ensure smooth, 
# safe motion along the planned path.
def follow_path(pos, heading, path, idx, lidar_data):
    """Follow path with look-ahead and smooth steering
    pos: robot’s current position (from GPS). 
    heading: current orientation.
    path: list of waypoints (map pixels).
    idx: current waypoint index.
    lidar_data: front/left/right distances 
    
    returns vx,vy, w """
    #There are no more points to follow
    
    if idx >= len(path):
        return 0, 0, 0, idx, True  

    # Emergency stop inside path-following (only if we haven't decided to dodge)
    if lidar_data['emergency']: #"emergency": front_min < EMERGENCY_DISTANCE
        return 0, 0, 0, idx, False

    # checking how far the Current waypoint is from where it is heading to 
    cx, cy = path[idx] #current waypoint 
    wx, wy = map_to_real(cx, cy) #convert to real world meters 
    dx = wx - pos[0] # how far from robot's x & y direction  
    dy = wy - pos[1]
    dist = math.hypot(dx, dy) #distance between two points

    # Waypoint reached, move to next
    if dist < WAYPOINT_REACH_DISTANCE:
        idx += 1 #next waypoint in the list
        if idx >= len(path):
            return 0, 0, 0, idx, True
        
        # Update to new waypoint
        cx, cy = path[idx]
        wx, wy = map_to_real(cx, cy)
        dx = wx - pos[0]
        dy = wy - pos[1]
        dist = math.hypot(dx, dy)

    # Look-ahead: if very close, target next waypoint
    LOOKAHEAD = 0.6
    if dist < LOOKAHEAD and idx + 1 < len(path):
        cx, cy = path[idx + 1] #for next waypoint
        wx, wy = map_to_real(cx, cy)
        dx = wx - pos[0]
        dy = wy - pos[1]

    # Target heading
    target_heading = math.atan2(dy, dx) #angle from robot to waypoint. dx = target_x - robot_x
    err = normalize_angle(target_heading - heading)

    # If heading error is large, rotate in place
    if abs(err) > 0.9:
        return 0.5, 0, 0.6 * err, idx, False

    # Speed control based on distance and heading error
    speed = min(MAX_SPEED, dist * 2.0) # properly aligned move fast 
    speed *= max(0.3, 1.0 - abs(err) * 0.8) # else move slowly but not less than speed <30%

    # Reduce speed near obstacles
    if lidar_data['front'] < SAFE_DISTANCE:
        safety_factor = (lidar_data['front'] - EMERGENCY_DISTANCE) / (SAFE_DISTANCE - EMERGENCY_DISTANCE)
        safety_factor = max(0.2, min(1.0, safety_factor))
        speed *= safety_factor

    # Velocity in robot frame
    vx = speed * math.cos(err) #err = 0° (cos(0)=1) , err = 90° (cos(90) =0) how much of the motion should be forward
    vy = speed * math.sin(err) #err = 0° (sin(0)=0) , err = 90° (sin(90) =1 ) how much of the motion should be sideways 
    w = 1.5 * err #bigger rotation if the err is bigger 

    return vx, vy, w, idx, False

#   GOAL POSITIONS (REAL WORLD METERS)
GOALS = {
    'duck':     (-4.43, -5.35),
    'chair':    (-2.5,  -2.55),
    'dog':      (-2.5,  1.96),
    'phone':    (-1.6,   5.24),
    'cat':      (-1.63, -5.46),
    'computer': ( 1.93,  5.42),
    'flower':   ( 2.45, -5.47),
    'ball':     ( 4.28,  2.73),
    'gnome':    ( 4.27, -2.26),
}
# =============================================================
#   OBSTACLE AVOIDANCE -Rotational Scanning AND Lateral Avoidance (Scanning + Emergency Dodge)
# =============================================================
"""The key behaviours implemented are:
    1. Real-Time LIDAR-Based Detection
        Distances from the LIDAR are continuously analysed to detect objects within:
            Emergency Distance: Immediate stop or reverse.
            Critical Distance: Trigger scanning behaviour.
            Safe Distance: Navigate normally.
    2. 360° Rotational Scanning
            When the robot encounters an unexpected obstacle, it performs:
                A full rotation,
                Computes the clearance in each direction,
                Selects the safest heading with maximum free space,
                Aligns itself and executes a controlled forward motion.
    3. Mecanum-Based Lateral Avoidance
        Mecanum wheels allow sideways motion without rotation.
I incorporated:
    Lateral "sidesteps"
    Continuous forward progress even during avoidance
    Smooth return to the planned path once clear
This choice was principled: mecanum locomotion provides manoeuvrability ideal for indoor, cluttered museum spaces, avoiding unnecessary rotations and enabling fluid motion"""

SCAN_TURN_SPEED = 0.8   # angular speed during scan (rad/s approx) , how fast 
SCAN_SWEEP_STEPS = 22   # how many steps to sweep each side
SCAN_ALIGN_STEPS = 15   # how long to turn towards best side
# Defines a small state machine used when the robot is blocked: it rotates left and right in controlled sweeps to measure which direction has the most free space.
scan_state = {
    "active": False,
    "phase": "IDLE",      # Phases : 'IDLE', 'SWEEP_LEFT', 'SWEEP_RIGHT', 'ALIGN'
    "step": 0,
    "best_dist": 0.0,
    "best_dir": 0,        # +1 = left, -1 = right, 0 = none
}
#The reset_scan() function clears this state so the robot can start a fresh scan whenever a new obstacle situation occurs.
def reset_scan():
    scan_state["active"] = False
    scan_state["phase"] = "IDLE"
    scan_state["step"] = 0
    scan_state["best_dist"] = 0.0
    scan_state["best_dir"] = 0


def lateral_scan(lidar_data):
    """
    Rotational scan:
    - Rotate left then right while tracking best front clearance.
    - Then align towards best direction.
    - If repeated scans fail, higher-level logic will force rotate.
    """

    #starts a rotational scan whenever the front LiDAR distance is too small and a clear path is needed.
    
    global scan_fail_count

    front = lidar_data["front"] #getting front distance data

    # Start scan only when needed
    if not scan_state["active"]:
        if front >= CRITICAL_DISTANCE:
            return None

        scan_state["active"] = True
        scan_state["phase"] = "SWEEP_LEFT"
        scan_state["step"] = 0
        scan_state["best_dist"] = front
        scan_state["best_dir"] = 0

    phase = scan_state["phase"]
    step = scan_state["step"]

    # PHASE 1: LEFT SWEEP 
    #It first sweeps left while keeping track of which direction gives the largest free space, and returns a turning command until the left sweep is complete.
    if phase == "SWEEP_LEFT":
        if front > scan_state["best_dist"]: #current lidar with best_dist
            scan_state["best_dist"] = front
            scan_state["best_dir"] = +1 #left

        scan_state["step"] += 1

        if step < SCAN_SWEEP_STEPS:
            return 0, 0, +SCAN_TURN_SPEED, "SCAN-ROTATE-LEFT" #rotate left (vx,vy,w, mode)
        else:
            scan_state["phase"] = "SWEEP_RIGHT"
            scan_state["step"] = 0
            return 0, 0, 0, "SCAN-SWITCH-RIGHT"

    # PHASE 2: RIGHT SWEEP 
    # performs the right-side sweep, checking whether turning right offers more clearance than the left sweep and updating the best escape direction
    if phase == "SWEEP_RIGHT":
        if front > scan_state["best_dist"]:
            scan_state["best_dist"] = front
            scan_state["best_dir"] = -1 #right

        scan_state["step"] += 1

        if step < SCAN_SWEEP_STEPS:
            return 0, 0, -SCAN_TURN_SPEED, "SCAN-ROTATE-RIGHT"
        else:
            scan_state["phase"] = "ALIGN"
            scan_state["step"] = 0
            return 0, 0, 0, "SCAN-DECIDE"

    #  PHASE 3: ALIGN 
    #Once the sweep completes, it transitions to the “ALIGN” phase so the robot can rotate toward the chosen safest direction.
    if phase == "ALIGN":
        scan_state["step"] += 1

        # If no good direction found, scan failed
        if scan_state["best_dir"] == 0:
            scan_fail_count += 1
            reset_scan()
            return None

        # Align towards best direction
        if step < SCAN_ALIGN_STEPS:
            w = SCAN_TURN_SPEED * scan_state["best_dir"]
            return 0, 0, w, "SCAN-ALIGN-BEST"
        else:
            # Scan successful → reset fail count
            scan_fail_count = 0
            reset_scan()
            return None

    return None


def lateral_avoidance(lidar_data):
    """
    SECOND BRAIN : INSTANT REACTION 
    Minimal lateral avoidance.
    - ONLY use mecanum side slip in true emergency to avoid collision.
    - For normal obstacles, rely on rotational scan + replanning.
    """
    front = lidar_data["front"]
    left = lidar_data["left"]
    right = lidar_data["right"]

    # True emergency: obstacle too close in front
    if front < EMERGENCY_DISTANCE:
        # quick small side hop
        if right > left:
            return 0, -1.0, 0, "RIGHT-EMERGENCY"
        else:
            return 0, +1.0, 0, "LEFT-EMERGENCY"

    # For non-emergency, let rotational scan + path following handle it
    return None

# =============================================================
#   NAVIGATION LOOP
# =============================================================

goal_queue = []
current_goal = None
path = None
wp_index = 0 #which step along the route we are currently aiming for
state = "IDLE"
replan_counter = 0

cv2.namedWindow("map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("map", int(W*1.2), int(H*1.2))

print("\n MUSEUM NAVIGATION ")
print("Available goals:", list(GOALS.keys()))
print("Type goal name to navigate.\n")


while robot.step(timestep) != -1:
    #  CAMERA FEED 
    if camera is not None:
        img = camera.getImage()
        if img:
            w_cam = camera.getWidth()
            h_cam = camera.getHeight()
            img_array = np.frombuffer(img, np.uint8).reshape((h_cam, w_cam, 4))
            frame = img_array[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("camera", frame)
            cv2.waitKey(1)

    pos = get_robot_position() # robots current position
    heading = get_robot_heading() # compass-based heading
    lidar_data = analyze_lidar(lidar.getRangeImage()) # processed LiDAR obstacle distances
    cmd = keyboard.get_command()

    # ---------------------------------------------------------
    # KEYBOARD INPUT
    # ---------------------------------------------------------

    #As required by the brief, multiple user commands must be handled through goal queuing.
    #Design principles behind this choice:
        #New commands must not erase or overwrite current tasks.
        #Users may request goals while the robot is in transit.
        #The executive layer processes the queue sequentially.
    if cmd:
        cmd = cmd.strip().lower()

        if cmd in GOALS:
            goal_queue.append(cmd)
            print(f"[QUEUED] {cmd} | Queue: {goal_queue}")

            if current_goal is None:
                current_goal = goal_queue.pop(0)
                state = "PLANNING"
        else:
            print(f"[UNKNOWN] '{cmd}' - Available: {list(GOALS.keys())}")

    # ---------------------------------------------------------
    # BEHAVIOUR COORDINATION STRATERGY (Finite State Machine)
    # ---------------------------------------------------------

    # ---------------- PLANNING ----------------
    # Reset scan state so previous scan behaviour doesn’t interfere
    if state == "PLANNING":
        reset_scan()

        if current_goal is None:# If there is no active goal, switch to IDLE and stop planning
            state = "IDLE"
            continue

        # Convert robot and goal positions from real-world meters to map pixels    
        goal_pos = GOALS[current_goal]
        start_px = real_to_map(pos)
        goal_px = real_to_map(goal_pos)

        # Run Dijkstra to compute a global path on the occupancy grid
        print(f"[PLANNING] Dijkstra path to '{current_goal}'...")
        raw_path = dijkstra(start_px, goal_px)

        # If no valid path is found, move to the next queued goal or go idle
        # If a valid path exists, simplify it for smoother navigation
        # Reset waypoint index and replanning timer for fresh navigation
        # Switch to NAVIGATING state to begin following the computed path
        if raw_path is None:
            print(f"[FAILED] No path to '{current_goal}'")
            
            if goal_queue: #check for other goal 
                current_goal = goal_queue.pop(0)
                state = "PLANNING"
            else:
                current_goal = None
                state = "IDLE"
        else:
            path = simplify_path(raw_path, step=8)
            wp_index = 0
            replan_counter = 0
            state = "NAVIGATING"
            print(f"[STARTING] Following path with {len(path)} waypoints")


        
    # ---------------- NAVIGATING ----------------
    # Compute distance from robot to the current goal
    elif state == "NAVIGATING":
        goal_pos = GOALS[current_goal]
        dist_goal = math.hypot(goal_pos[0] - pos[0], goal_pos[1] - pos[1])

        # Check if the robot has reached the goal position
        if dist_goal < GOAL_REACH_DISTANCE:
            print(f"[REACHED] Goal '{current_goal}' ")
            path = None # If reached, stop movement and clear the current path
            reset_scan() # Reset scan state to avoid leftover avoidance behaviour
            mecanum(0, 0, 0)

            # If more goals are queued, load the next goal and switch to PLANNING
            # If no goals remain, switch to IDLE and stop navigation
            if goal_queue:
                current_goal = goal_queue.pop(0)
                state = "PLANNING"
                print(f"[NEXT] Moving to '{current_goal}' | Remaining: {goal_queue}")
            else:
                current_goal = None
                state = "IDLE"
                print("[IDLE] All goals completed")
            continue

        # Periodic replanning for dynamic obstacles
        replan_counter += 1
        if replan_counter >= REPLAN_INTERVAL:
            print("[REPLAN] Updating path...")
            replan_counter = 0
            state = "PLANNING"
            continue

        # ---------- TOO MANY FAILED SCANS → FORCE ROTATE ----------
        # If repeated scans fail, force the robot to rotate in place to escape trapped situations  
        # After rotating, reset the scan counter and trigger a fresh global replan  
        if scan_fail_count >= SCAN_FAIL_LIMIT:        
            mecanum(0, 0, 1.2)
            for _ in range(30):
                robot.step(timestep)
            scan_fail_count = 0
            state = "PLANNING"
            continue
        
        # ---------- ROTATIONAL SCAN ----------
        if lidar_data["front"] < CRITICAL_DISTANCE:
            mecanum(-0.25, -0.2, -0.2)   # reverse a bit
            for _ in range(20):    
                robot.step(timestep)
            mecanum(0, 0, 0)
            
         # wait for 20 step to decide 
            for _ in range(20):          
                robot.step(timestep)
        
            # Start rotational scan to determine safest direction
            scan_cmd = lateral_scan(lidar_data)
        
        # ---------- START SCAN AFTER BACKUP ---------
        else:
            scan_cmd = None 
            
        if scan_cmd is not None: #move in the safest side 
            vx, vy, w, mode = scan_cmd
            mecanum(vx, vy, w)
            continue

        # ===== SCAN FINISHED → WAIT BEFORE NEXT ACTION =====
        # After a scan completes, pause briefly to stabilize before moving again  
        # Then trigger a replan so the robot navigates using its newly aligned heading  
        if scan_state["active"] == False and scan_cmd is None and lidar_data["front"] < CRITICAL_DISTANCE:
            mecanum(0, 0, 0)
            for _ in range(12):   # 12 timesteps = ~250–300ms depending on timestep
                robot.step(timestep)
            # after waiting, replan with new heading
            state = "PLANNING"
            continue

       # ---------- CORNER ESCAPE ----------
       # Detect when the robot is stuck in a tight front-left or front-right corner based on LiDAR readings  
       # Perform a diagonal backward escape move with a slight rotation to free the robot from the corner  
       # Stop motion after the escape to regain stability before continuing navigation  
        if lidar_data["front"] < 0.9 and lidar_data["left"] < 1.0:
            for _ in range(6):
                mecanum(-1.0, -0.5, -0.2)  # back + right + rotate
                robot.step(timestep)
            mecanum(0,0,0)
        
        elif lidar_data["front"] < 0.9 and lidar_data["right"] < 1.0:
            for _ in range(6):
                mecanum(-1.0, +0.5, +0.2)  # back + left + rotate
                robot.step(timestep)
            mecanum(0,0,0)


        # ---------- MINIMAL LATERAL AVOIDANCE (EMERGENCY ONLY) ----------
        # Check for emergency lateral avoidance and execute an instant sideways dodge if needed  
        # Apply the dodge velocities and skip normal navigation for this timestep to ensure collision prevention  
        avoid = lateral_avoidance(lidar_data)
        if avoid is not None:
            vx, vy, w, mode = avoid
            mecanum(vx, vy, w)
            print(f"[DODGE] Mode={mode}")
            continue

        # ---------- NORMAL PATH FOLLOWING ----------
        vx, vy, w, wp_index, done = follow_path(pos, heading, path, wp_index, lidar_data)
        if done:
            # Path completed but not at goal yet - replan
            state = "PLANNING"
        else:
            mecanum(vx, vy, w)

    # ---------------- IDLE ----------------
    else:
        reset_scan()
        mecanum(0, 0, 0)

    # =============================================================
    #   VISUALIZATION
    # =============================================================
   
    disp = cv2.cvtColor((occupancy_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw path
    if path:
        for i in range(len(path) - 1):
            cv2.line(disp, path[i], path[i + 1], (0, 255, 0), 2)
        
        # Current target waypoint
        if wp_index < len(path):
            cv2.circle(disp, path[wp_index], 5, (0, 255, 255), -1)

    # Draw robot
    rx, ry = real_to_map(pos)
    cv2.circle(disp, (rx, ry), 10, (255, 0, 0), -1)

    # Heading arrow
    hx = int(rx + 20 * math.sin(heading))
    hy = int(ry - 20 * math.cos(heading))
    cv2.arrowedLine(disp, (rx, ry), (hx, hy), (255, 255, 0), 2, tipLength=0.3)   

    cv2.imshow("map", disp)
    cv2.waitKey(1)

#The solution is built on principled design grounded in robotics fundamentals:
    #A hybrid architecture integrating planning and reactivity
    #Grid-based mapping and shortest-path planning
    #Real-time LIDAR scanning and obstacle avoidance
    #Mecanum-based motion control
    #Multi-goal queuing via behaviour coordination
#The result is a system that can navigate a museum environment efficiently, and robustly,
#With behaviour that reflects deliberate design choices rather than hand-crafted patches.