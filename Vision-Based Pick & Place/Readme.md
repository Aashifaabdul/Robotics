# 🦾 YouBot – Autonomous Table Cleaning with Vision-Based Pick & Place

This project implements an **autonomous mobile manipulation system** using a **KUKA YouBot** in the **Webots simulator**.  
The robot navigates to a table, **detects objects using vision and depth**, picks them up with a gripper, and **places them into a bin**, repeating the process until the workspace is clean.

The system integrates **navigation, perception, manipulation, and behaviour coordination** into a single autonomous pipeline.

---
<img width="1417" height="714" alt="image" src="https://github.com/user-attachments/assets/e0e07fd9-62ad-4cc3-bbe1-8e918889dd3a" />
<img width="1132" height="705" alt="image" src="https://github.com/user-attachments/assets/e1822edc-57d6-4556-a341-7588aab76953" />


## 🎯 Task Overview

**Scenario:**  
A mobile manipulator cleans a table by:
1. Navigating to predefined table locations
2. Detecting cubes using RGB + depth sensing
3. Grasping objects autonomously
4. Transporting them to a bin
5. Repeating until no objects remain

This mirrors real-world service robotics tasks such as **cleaning, sorting, and assistive manipulation**.

---

## 🧠 System Architecture

The solution is structured into **three tightly coordinated subsystems**:

### 1️⃣ Navigation (Mobile Base)
- GPS + Compass for localisation
- Two-step navigation strategy:
  - **Rotate-to-target**
  - **Move-straight-with-heading-correction**
- Fixed navigation targets:
  - `TABLE_POS`
  - `WAYPOINT1`
  - `BIN_STOP_POS`
- Consistent base alignment for repeatable perception

---

### 2️⃣ Perception (Vision + Depth Fusion)
- **RGB Camera (Recognition enabled)**
- **Depth Camera**
- Object detection using Webots recognition
- 2D → 3D back-projection using:
  - Camera intrinsics
  - Depth image
  - Forward kinematics
- Closest valid cube is selected for manipulation
- Continuous visual feedback with overlays

---

### 3️⃣ Manipulation (Pick & Place)
- Joint-space control (`move_PTP`) for safe arm configurations
- Cartesian control (`move_LIN`) for precise vertical lifting
- Two-phase visual servoing strategy:
  - **Phase 1:** Rotate base to horizontally align object with TCP
  - **Phase 2:** Continuous approach with pixel + 3D correction
- Controlled gripper closing and lift
- Deterministic bin placement using predefined arm poses

---

## 📁 File Structure

```
youbot.py        # High-level autonomous behaviour (navigation + vision + manipulation)
youbot_base.py   # Robot abstraction, kinematics, motion primitives
```

### `youbot_base.py`
Provides:
- Mecanum base control
- Joint and gripper control
- Forward & inverse kinematics (KinPy)
- Linear and point-to-point arm motion
- Sensor initialisation (camera, depth, GPS, compass)

### `youbot.py`
Implements:
- Autonomous cleaning logic
- Vision-based object detection
- Visual servoing for grasping
- Navigation between table, waypoint, and bin
- Pick–carry–place task loop

---

## 🔄 Behaviour Flow

1. Initialise robot and arm in safe pose
2. Navigate to table center
3. Detect closest cube
4. Align cube with gripper using vision
5. Approach and grasp object
6. Lift and move to carry pose
7. Navigate to bin
8. Place cube in bin
9. Repeat for multiple cycles and waypoints

The system automatically terminates when no more objects are detected.

---

## ⚙️ Requirements

- **Webots Simulator**
- YouBot model with:
  - RGB camera (recognition enabled)
  - Depth camera
  - GPS
  - Compass
  - Mecanum wheels
  - Gripper arm
- Python libraries:
  - `numpy`
  - `opencv-python`
  - `kinpy`
  - `scipy`

---

## ▶️ How to Run

1. Open the Webots world containing the **YouBot**
2. Place `youbot.py` and `youbot_base.py` in the controller folder
3. Assign `youbot.py` as the controller
4. Start the simulation

The robot will:
- Clean the table autonomously
- Display live camera and detection overlays
- Print task progress to the console

---
