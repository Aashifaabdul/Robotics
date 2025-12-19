import math
import numpy as np
import cv2 as cv
from controller import Robot


# ===============================
# PERCEPTION: SENSORS & DATA PROCESSING
# ===============================


# ROBOT SETUP
robot = Robot()
timestep = int(robot.getBasicTimeStep())  # Webots simulation step (ms)


# ---- Sensor Initialization ----
camera = robot.getDevice('camera')
camera.enable(timestep)
width = camera.getWidth()
height = camera.getHeight()

rangefinder = robot.getDevice('range-finder')
rangefinder.enable(timestep)


def detect_black_line(image):
    """
    Detect the black line on the floor using the camera.
    RATIONALE:
      - The robot follows a black line on a lighter background.
      - We convert to grayscale + threshold to isolate the line.
      - We only look at the BOTTOM HALF of the image (where the ground is).
      - We split that region into LEFT, CENTER, RIGHT bands and measure how much line is present.
      - These three values are used to compute an error for PID line-following.
    Returns:
      left, center, right intensity (higher = more black line in that region), binary image.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #pixel <70 = dark : invert - white , 
    # pixel >70 = white : invert - black  
    _, binary = cv.threshold(gray, 70, 255, cv.THRESH_BINARY_INV)
    
    h, w = binary.shape
    # Use only bottom half as Region Of Interest (ROI) near the robot
    roi = binary[int(h*0.5):, :]

    third = w // 3
    # Higher the mean → more white pixels → more space on that side 
    left = np.mean(roi[:, :third])
    center = np.mean(roi[:, third:2*third])
    right = np.mean(roi[:, 2*third:])
    
    return left, center, right, binary


def analyze_free_space(image, rf):
    """
    Analyze both left and right sides to determine which side has more free space
    when avoiding an obstacle.
    
    RATIONALE:
      - When an obstacle is detected, we don't blindly pick left or right.
      - We use the RANGEFINDER to estimate how much space exists on each side.
      - We also check RED areas (obstacles) on each side from the camera.
      - Then we compute a score: more distance + less red = better path.
      - This helps choose a safer avoidance direction instead of random turning.
      
    Returns:
      direction: 'LEFT' or 'RIGHT' (side with better escape route)
      confidence: how strongly one side is better than the other
      left_space, right_space: average free space on each side
      left_red, right_red: amount of red pixels (obstacles) on each side
    """
    rf_data = rf.getRangeImage()
    if not rf_data:  # no range data → fall back to arbitrary choice
        return 'LEFT', 0.5, 0.0, 0.0, 0, 0 #random direction , confidence , left space , right space, left red, right red 
    
    distances = np.array(rf_data)
    rf_width = len(distances)

    # Split range scan into left and right halves
    left_half = distances[:rf_width//2]
    right_half = distances[rf_width//2:]
    
    # Filter out invalid readings (NaN, inf, or too small distance)
    left_valid = left_half[(np.isfinite(left_half)) & (left_half > 0.1)]
    right_valid = right_half[(np.isfinite(right_half)) & (right_half > 0.1)]
    
    # Average distance on each side = how open that direction is
    left_space = np.mean(left_valid) if len(left_valid) > 0 else 0
    right_space = np.mean(right_valid) if len(right_valid) > 0 else 0
    
    # ---- Detect red blobs (obstacles) in camera on each side ----
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask2 = cv.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_mask = cv.bitwise_or(mask1, mask2)
    
    h, w = red_mask.shape
    # Ignore top 30% of image: far away / background, focus on floor region near robot
    red_mask[:int(h*0.3), :] = 0
    
    # Count how much red is seen on each half
    left_red = np.sum(red_mask[:, :w//2] > 0)
    right_red = np.sum(red_mask[:, w//2:] > 0)
    
    # ---- Scoring: more space, less red = better direction ----
    # Distance is scaled up, red counts are penalized slightly
    left_score = left_space * 10 - left_red * 0.01
    right_score = right_space * 10 - right_red * 0.01
    
    # If one side has significantly less red, give it a small bonus
    if abs(left_red - right_red) > 300:
        if left_red < right_red:
            left_score += 0.3
        else:
            right_score += 0.3
    
    # Choose direction based on higher score and compute a confidence
    if left_score > right_score:
        direction = 'LEFT'
        confidence = min(1.0, (left_score - right_score) / max(1, abs(right_score))) # confidence stays btw 0-1 
    else:
        direction = 'RIGHT'
        confidence = min(1.0, (right_score - left_score) / max(1, abs(left_score)))
    
    return direction, confidence, left_space, right_space, left_red, right_red


def detect_red_obstacle(image):
    """
    Detect whether a red obstacle is present in front of the robot, using the camera.
    
    RATIONALE:
      - Obstacles are marked with red color.
      - We detect red pixels in the lower part of the image (ground area).
      - We also visualize the detection for debugging.
      - If red pixels exceed a threshold, we say 'an obstacle is present'.
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Two HSV ranges to capture red (wrapped around hue boundaries)
    mask1 = cv.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])) #lower red hue tones 
    mask2 = cv.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255])) #uppper red hue tones 
    red_mask = cv.bitwise_or(mask1, mask2) #white pixel - red detected , black pixel - not red 
    
    h, w = red_mask.shape
    # Ignore top 30%: focus on floor-level region
    red_mask[:int(h*0.3), :] = 0 
    
    # Visualization for understanding what the robot "sees"
    vis_image = image.copy()
    cv.rectangle(vis_image, (0, int(h*0.3)), (w//2, h), (255, 255, 0), 2)
    cv.rectangle(vis_image, (w//2, int(h*0.3)), (w, h), (0, 255, 255), 2)
    
    red_overlay = cv.cvtColor(red_mask, cv.COLOR_GRAY2BGR) #red mask (grayscale )[1 chanel] - BGR (3 chanel) 
    vis_image = cv.addWeighted(vis_image, 0.7, red_overlay, 0.3, 0) #70% of original img and 30% red mask - the glow 
    
    red_pixels = np.sum(red_mask > 0)
    cv.putText(vis_image, f"Red Pixels: {red_pixels}", (10, 20), 
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv.imshow("Camera View", vis_image)
    cv.waitKey(1)
    
    
    # Return whether obstacle exists and how many red pixels were seen
    return red_pixels > RED_THRESHOLD, red_pixels


def get_obstacle_distance(rf):
    """
    Estimate the distance to the closest obstacle using the rangefinder.
    
    RATIONALE:
      - Range images can be noisy.
      - We filter invalid / tiny readings, sort them, take the few smallest,
        and use their median as a robust estimate of nearest obstacle distance.
      - If nothing valid is found, we return infinity (no obstacle detected).
    """
    try:
        rf_data = rf.getRangeImage()
        if rf_data:
            distances = np.array(rf_data)
            valid_distances = distances[(np.isfinite(distances)) & (distances > 0.1)]
            if len(valid_distances) > 0:
                sorted_dist = np.sort(valid_distances)
                closest_few = sorted_dist[:min(3, len(sorted_dist))] #get first 3 shortest distance 
                return np.median(closest_few) # get the mid value to avoid outlier 
    except:
        pass
    return float('inf')


# ===============================
# CONTROL: ACTUATORS & ALGORITHMS
# ===============================


# ---- Motor Actuators ----
l_motor = robot.getDevice("left_wheel_hinge")
r_motor = robot.getDevice("right_wheel_hinge")
# Set to velocity control mode
l_motor.setPosition(math.inf)
r_motor.setPosition(math.inf)


# PID controller implementation
class PID:
    """
    Generic PID controller for 1D error correction.
    
    RATIONALE:
      - We use PID to smoothly correct the robot's deviation from the line.
      - P: reacts to current error (how far from the line now).
      - I: reacts to accumulated error.
      - D: reacts to change in error (prevents oscillation).
      - The output is a steering command that we apply differentially
        to left/right wheels.
    """
    def __init__(self, kp, ki, kd, out_limits=(-1.0, 1.0), integral_limits=(-float('inf'), float('inf'))):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min, self.out_max = out_limits
        self.int_min, self.int_max = integral_limits
        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        """Reset history when switching behavior modes (e.g., after avoidance)."""
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        """
        Compute new PID output for a given error and time step.
        
        RATIONALE:
          - Integrate error over time for I term.
          - Differentiate error over time for D term.
          - Combine P, I, D contributions and clamp to safe range.
        """
        # Integral term accumulates error across time (long-term bias correction)
        self.integral += error * dt 
        self.integral = max(self.int_min, min(self.int_max, self.integral))

        # Derivative term: how quickly error is changing (anticipates overshoot)
        derivative = 0.0 if dt == 0 else (error - self.prev_error) / dt
        
        # PID formula
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        # Clamp output to avoid extreme steering commands
        out = max(self.out_min, min(self.out_max, out))

        self.prev_error = error
        return out


def clamp(v, a, b):
    """
    Clamp a value between [a, b].
    
    RATIONALE:
      - Used for wheel speeds and steering to prevent unstable, extreme values.
      - Ensures commands stay within physical / safe limits.
    """
    return max(a, min(b, v))


# ---- PID Parameters setup ----
Kp = 1.0 #Main steering force
Ki = 0.003 #Slowly corrects bias
Kd = 0.5 #Smooths turns, prevents overshoot
PID_OUTPUT_LIMIT = 0.7  # Max absolute steering value


# Instantiate PID for line-following steering
pid = PID(Kp, Ki, Kd, out_limits=(-PID_OUTPUT_LIMIT, PID_OUTPUT_LIMIT),
          integral_limits=(-2.0, 2.0))


# ===============================
# BEHAVIOR: STATE MACHINE & MAIN LOGIC
# ===============================


# ---- Motion / Safety Parameters ----
MAX_SPEED = 15.0       # Robot’s maximum wheel speed (safety cap)
NORMAL_SPEED = 10.0    # Default line-following speed
RED_THRESHOLD = 200    # Minimum red pixels to consider 'obstacle present'
DANGER_DISTANCE = 1.2  # Distance threshold to trigger avoidance
SAFE_DISTANCE = 1.3    # Distance threshold to resume normal operation
MIN_RED_FOR_AVOIDANCE = 190  # Additional red threshold for avoidance


# ---- State tracking variables ----
avoiding = False          # Whether the robot is currently in avoidance mode
avoid_counter = 0         # How many control cycles spent in avoidance
chosen_direction = None   # 'LEFT' or 'RIGHT' chosen for avoidance path
AVOID_TIME = 80           # Max number of steps to stay in avoidance trajectory


# MAIN LOOP
print(" Starting JetBot")

prev_time = robot.getTime()

while robot.step(timestep) != -1:
    # dt controls how the PID derivative & integral work (time-based)
    cur_time = robot.getTime()
    dt = cur_time - prev_time if cur_time - prev_time > 0 else (timestep / 1000.0) #If time did not move forward (dt=0), assume dt = simulation timestep
    prev_time = cur_time

    # ===================
    # 1. PERCEPTION STEP
    # ===================
    img = camera.getImage()
    image = np.frombuffer(img, np.uint8).reshape((height, width, 4))
    bgr = image[:, :, :3].copy()

    left, center, right, binary = detect_black_line(bgr)
    cv.imshow("Binary Line Detection", binary)
    has_obstacle, red_count = detect_red_obstacle(bgr)
    distance = get_obstacle_distance(rangefinder)

    # Decide if an obstacle is both close and visually confirmed
    obstacle_is_close = (
        has_obstacle and distance < DANGER_DISTANCE and red_count > MIN_RED_FOR_AVOIDANCE
    )
    
    # ===========================
    # 2. BEHAVIORAL LOGIC (STATE MACHINE)
    # ===========================
    
    # ---------- STATE 1 : NORMAL LINE-FOLLOWING MODE ----------
    if not avoiding:
        # Compute line-following error from left/right intensities
        # total prevents division by zero; pos > 0 means line more to the right, pos < 0 → more to the left
        total = left + center + right + 1e-6 
        pos = (right - left) / total
        error = pos #how far i am from the centre of line 

        # Adapt PID gain: if something is close, reduce Kp for smoother behavior
        pid.kp = 0.7 if distance < 1.0 else 1.2
        steering = pid.update(error, dt)
        steering = clamp(steering, -PID_OUTPUT_LIMIT, PID_OUTPUT_LIMIT)

        # Slow down when an obstacle is nearby to be safer & more stable
        speed_factor = 0.7 if (has_obstacle and distance < SAFE_DISTANCE) else 1.0
        if red_count > 200:
            speed_factor = 0.5  # Stronger slow-down if lots of red visible
        base_speed = NORMAL_SPEED * speed_factor
        
        # Differential steering: steering alters left/right speeds to turn towards the line
        #steering > 0 → line is on the right
        #steering < 0 → line is on the left
        #steering = 0 → perfectly centered
        l_speed = clamp(base_speed * (1.0 + steering), -MAX_SPEED, MAX_SPEED)
        r_speed = clamp(base_speed * (1.0 - steering), -MAX_SPEED, MAX_SPEED)

        # ---- Transition to Avoidance Mode ----
        if obstacle_is_close:
            print(f"\nOBSTACLE DETECTED at the Distance: {distance:.2f} m ")
            
            # Stop the robot briefly to stabilize before avoidance
            l_motor.setVelocity(0)
            r_motor.setVelocity(0)
            wait_steps = int((0.3 * 1000) / timestep)  # Wait ~0.3 seconds
            for _ in range(wait_steps):
                robot.step(timestep)
            
            # Use range + vision to choose side with more space and less red
            direction, conf, l_space, r_space, l_red, r_red = analyze_free_space(bgr, rangefinder)
            print(f"Direction: {direction}")
            
            # Gentle reverse to create space from the obstacle before turning
            for _ in range(6):
                l_motor.setVelocity(-NORMAL_SPEED * 0.3)
                r_motor.setVelocity(-NORMAL_SPEED * 0.3)
                robot.step(timestep)
            
            # Switch to avoidance mode
            avoiding = True
            avoid_counter = 0
            chosen_direction = direction
            pid.reset()  # Reset PID so old line-following error doesn't influence post-avoidance behavior
            continue

    # ---------- STATE 2 : OBSTACLE AVOIDANCE MODE ----------
    else:
        avoid_counter += 1
        base_speed = NORMAL_SPEED * 0.8  # Slightly slower, more controlled
        
        # Early exit condition:
        # If obstacle is now far enough and we've spent some time avoiding, go back to normal
        if distance > SAFE_DISTANCE and avoid_counter > 35:
            avoiding = False
            avoid_counter = 0
            pid.reset()
            print(f"Obstacle cleared")
            continue

        # Smooth avoidance logic based on chosen direction.
        # RATIONALE:
        #   The motion is time-shaped:
        #   - First phase: strong turning to move away from obstacle quickly.
        #   - Second phase: smoother curvature to go around it.
        #   - Third phase: easing back towards straight to re-align.
        if chosen_direction == "LEFT":
            # We want to avoid to the LEFT side → turn the robot to the RIGHT initially.
            if avoid_counter < 25:
                # Phase 1: Hard right turn (bigger speed difference)
                l_speed, r_speed = base_speed * 0.6, base_speed * 0.95
            elif avoid_counter < 55:
                # Phase 2: Smooth right turn to continue curving around
                l_speed, r_speed = base_speed * 0.8, base_speed * 0.9
            elif avoid_counter < AVOID_TIME:
                # Phase 3: Ease back towards straight
                l_speed, r_speed = base_speed * 0.9, base_speed * 0.65
            else:
                # Safety exit if we somehow stayed in avoidance too long
                avoiding = False
                pid.reset()
                
        else:  # chosen_direction == "RIGHT"
            # We want to avoid to the RIGHT side → turn the robot to the LEFT initially.
            if avoid_counter < 25:
                # Phase 1: Hard left turn
                l_speed, r_speed = base_speed * 0.95, base_speed * 0.6
            elif avoid_counter < 55:
                # Phase 2: Smooth left turn
                l_speed, r_speed = base_speed * 0.9, base_speed * 0.8
            elif avoid_counter < AVOID_TIME:
                # Phase 3: Ease back towards straight
                l_speed, r_speed = base_speed * 0.65, base_speed * 0.9
            else:
                # Safety exit
                avoiding = False
                pid.reset()
                

    # ===================
    # 3. CONTROL STEP: Apply Output
    # ===================
    # Final clamping ensures wheel commands are always within physical limits
    l_motor.setVelocity(float(clamp(l_speed, -MAX_SPEED, MAX_SPEED)))
    r_motor.setVelocity(float(clamp(r_speed, -MAX_SPEED, MAX_SPEED)))
