# 🏛️ Summit XL – Autonomous Museum Navigation System

This project implements a **full autonomous navigation system** for the **Summit XL mobile robot** in **Webots**, designed for indoor environments such as museums.

The system combines **global path planning**, **local waypoint tracking**, and **reactive obstacle avoidance** using a **hybrid deliberative–reactive architecture**, enabling the robot to safely navigate to multiple user-specified goals in dynamic environments.

---
<img width="1915" height="989" alt="image" src="https://github.com/user-attachments/assets/e29565f7-3e22-44db-b78d-cf306c179eee" />

## 🎯 Key Capabilities

- **Multi-goal navigation** via keyboard input
- **Global path planning using Dijkstra**
- **Grid-based occupancy map with obstacle inflation**
- **Local waypoint following with mecanum kinematics**
- **Real-time LiDAR-based obstacle avoidance**
- **Rotational scanning for safe path recovery**
- **Emergency lateral dodging using mecanum wheels**
- **Finite State Machine (FSM) for behaviour coordination**
- **Live visualisation of map, path, robot pose, and heading**

---

## 🧠 System Architecture

The system follows a **three-layer robotics architecture**:

### 1️⃣ Deliberative Layer – Global Navigation
- Uses an **occupancy grid map**
- Computes collision-free paths using **Dijkstra’s algorithm**
- Inflates obstacles to maintain safety margins
- Supports **replanning** when new goals are issued or paths become blocked

### 2️⃣ Reactive Layer – Obstacle Avoidance
- Processes **LiDAR data** in real time
- Detects critical and emergency obstacles
- Performs:
  - Rotational scanning to find free space
  - Lateral mecanum sidesteps in emergencies
- Prevents deadlocks and corner traps

### 3️⃣ Executive Layer – Behaviour Coordination
- Finite State Machine with states:
  - `IDLE`
  - `PLANNING`
  - `NAVIGATING`
- Manages:
  - Goal queue
  - Scan recovery
  - Path completion
  - Safe transitions between behaviours

---

## 📁 Files Overview

```
summit_xl.py        # Main navigation, planning, avoidance, FSM
keyboardreader.py  # Keyboard-based goal input handler
```

### `keyboardreader.py`
- Reads user input via Webots keyboard
- Supports queued goal commands
- Filters invalid goals
- Allows interactive navigation during execution

### `summit_xl.py`
Implements:
- Sensor fusion (LiDAR, GPS, Compass, Camera)
- Occupancy grid mapping and inflation
- Dijkstra global path planning
- Waypoint simplification
- Mecanum drive control
- Obstacle avoidance and recovery
- Live visualisation and debugging

---

## 🗺️ Navigation Flow

1. User enters a goal name (e.g. `duck`, `chair`, `computer`)
2. Goal is added to a **queue**
3. Robot plans a global path using Dijkstra
4. Path is simplified and followed waypoint-by-waypoint
5. Dynamic obstacles trigger:
   - Rotational scans
   - Local avoidance
   - Emergency lateral dodges if required
6. On reaching the goal:
   - Next queued goal is processed
   - Or robot returns to idle

---

## ⚙️ Requirements

- **Webots Simulator**
- Summit XL robot model with:
  - LiDAR (Hokuyo)
  - GPS
  - Compass
  - RGB Camera
  - Mecanum wheels
- Python libraries:
  - `numpy`
  - `opencv-python`

---

## ▶️ How to Run

1. Open the Webots world containing the **Summit XL robot**
2. Assign `summit_xl.py` as the robot controller
3. Ensure `keyboardreader.py` is in the same controller directory
4. Start the simulation
5. Type goal names in the Webots keyboard window and press **Enter**

Example goals:
```
duck, chair, dog, phone, cat, computer, flower, ball, gnome
```

---

## 📊 Visualisation

The system provides real-time visual feedback:
- Occupancy grid map
- Planned global path
- Current waypoint
- Robot position and heading
- Live camera feed

These visualisations aid debugging and explainability.

---

