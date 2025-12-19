# 🤖 Line Following & Obstacle Avoidance Robot (Webots)

This project implements a **vision-based line-following robot with intelligent obstacle avoidance** in the **Webots simulator**.  
The robot uses **camera vision**, a **rangefinder**, and a **PID controller** to follow a black line while safely detecting and avoiding red obstacles in its path.

---
<img width="1910" height="995" alt="image" src="https://github.com/user-attachments/assets/5a4db5fd-419a-4ad6-a529-48425f9cd5d5" />

## 🎯 Key Features

- **Black line detection** using camera vision and image thresholding
- **PID-based line following** for smooth and stable motion
- **Red obstacle detection** using HSV colour segmentation
- **Rangefinder-based distance estimation**
- **Intelligent obstacle avoidance**
  - Chooses left or right path based on free space and obstacle density
- **Finite State Machine (FSM)**
  - Normal line-following mode
  - Obstacle avoidance mode
- **Real-time visual debugging**
  - Binary line mask
  - Obstacle overlays
  - Red pixel count

---

## 🧠 System Architecture

### 1️⃣ Perception
- **Camera**
  - Detects black line on the floor
  - Detects red obstacles using HSV filtering
- **Rangefinder**
  - Estimates distance to nearby obstacles
  - Filters noisy readings for robustness

### 2️⃣ Control
- **PID Controller**
  - Proportional (P): corrects immediate line deviation
  - Integral (I): removes long-term bias
  - Derivative (D): reduces oscillations
- **Differential wheel control**
  - Steering achieved by varying left/right wheel velocities

### 3️⃣ Behaviour (State Machine)
- **Normal Mode**
  - Follows the black line
  - Adjusts speed when obstacles are nearby
- **Avoidance Mode**
  - Triggered when a red obstacle is close
  - Chooses safer direction (LEFT / RIGHT)
  - Executes a smooth multi-phase avoidance trajectory
  - Returns to line following once safe

---

## 📁 File Overview

```
jetson_py.py
```

**Main components inside the file:**
- Line detection (`detect_black_line`)
- Obstacle detection (`detect_red_obstacle`)
- Free-space analysis (`analyze_free_space`)
- PID controller class
- Obstacle avoidance state machine
- Motor control logic

---

## ⚙️ Requirements

- **Webots** (robot simulator)
- **Python controller enabled**
- Python libraries:
  - `numpy`
  - `opencv-python`

---

## ▶️ How to Run

1. Open your **Webots world** containing:
   - A robot with:
     - Camera
     - Rangefinder
     - Differential drive wheels
2. Assign `jetson_py.py` as the robot controller
3. Start the simulation



