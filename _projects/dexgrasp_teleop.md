---
layout: page
title: Dexhand AR Teleoperation
description: Apple Vision Pro teleoperation of a 7-DoF Rokae AR5 arm and an Inspire dexterous hand, with closed-form redundant IK and servoJ streaming at 8 ms.
img: assets/img/dexhand_teleop/cover.jpg
importance: 0
category: Industrial
toc:
  sidebar: left
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="assets/video/dexhand_teleop/servoj_final.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
</div>
<div class="caption">
    The final system: servoJ streaming with filtering and interpolation.
</div>

## Overview

This is the teleoperation front end of DexGraspAnything (v0.5). An operator wearing an Apple Vision Pro drives a 7-DoF Rokae AR5 arm and an Inspire six-actuator dexterous hand:

- The **left hand** moves the arm's end-effector.
- A **right-hand pinch** is the clutch. The arm moves only while the pinch is held and stops as soon as it is released.
- All **five fingers** are tracked independently and retargeted to the hand.

Every 8 ms the system reads the headset's wrist pose, filters it, computes the arm's target pose, solves inverse kinematics and sends joint angles to the controller. The controller interpolates each 8 ms step into 1 ms motion. End-to-end latency is about 72 ms, 50 ms of which is a deliberate buffer that buys smooth motion.

In a 275-second task session the system had **0 IK failures in 25,589 solves**, a **1.9 mm tracking residual**, IK times of **3.46 ms median and 5.96 ms p95**, and the hand ran at **20.4 Hz with no errors**.

## Why servoJ

The controller needs a joint command every millisecond, and Python cannot meet that deadline reliably. The Rokae SDK offers three ways to stream motion, and only one suits teleoperation:

| Interface | Semantics | Teleop latency |
| --- | --- | --- |
| MoveL | Each point is a task that must finish; it cannot be withdrawn | 240–500 ms, worse with faster hands |
| RT Cartesian | One pose per millisecond | No motion on this AR5 |
| **servoJ** | One joint target every 8 ms; each new target replaces the old one | **24 ms** |

The difference is the queue. MoveL points wait in line, so when the hand stops the arm still has to finish the queue. servoJ has no queue and always chases the latest target. The cost is that servoJ takes joint angles rather than poses, so the inverse kinematics has to be solved on our side.

## Pipeline

### Filtering

Headset poses jitter, and without filtering that noise goes straight into joint commands. Three stages handle it:

- A **Kalman filter** with a constant-velocity model suppresses noise and estimates hand velocity for later prediction.
- A **One-Euro filter** adapts its cutoff to speed. It holds still hands steady and adds almost no lag during fast motion, which a fixed low-pass filter cannot do.
- A **final filter before IK** cleans the target after prediction and frame mapping. Orientation is filtered on the first two columns of the rotation matrix and rebuilt with Gram–Schmidt, because filtering all nine entries separately does not give an orthogonal matrix.

### Target pose

When the clutch engages, the system records the hand pose and the end-effector pose as a reference. After that, the target is the reference plus the scaled, frame-rotated hand displacement. Because the reference is re-recorded at every engagement, the operator can "row": release, move the hand back to a comfortable spot, and engage again to keep going.

The headset streams at 60 Hz while the control loop runs at 125 Hz, so the Kalman velocity is used to extrapolate the hand position to the current control tick. The target is then clamped into a workspace box shrunk slightly inward from the arm's reach.

### Inverse kinematics

The AR5 has seven joints, but an end-effector pose fixes only six quantities. The extra degree of freedom is the **arm angle**, how far the elbow swings around the shoulder–wrist line, so every pose has infinitely many joint solutions. The solver works in two layers:

- **Closed-form solutions.** For a given arm angle, the remaining six joints are solved in closed form with the product-of-exponentials formulation and Paden–Kahan subproblems, with no iteration. Sweeping the arm angle yields a set of exact candidates in about 1.9 ms.
- **Selection.** Three filters pick one candidate:
  1. A **jump gate** discards candidates more than 0.25 rad (14.3°) from the previous solution.
  2. A **configuration filter** keeps only candidates in the same configuration as at engagement. Three bits (wrist flip, elbow side, shoulder front or back) give eight configurations, and crossing between them requires passing through a singularity, so the arm never switches posture by surprise.
  3. A **weighted cost** over continuity, joint-limit margin, distance to singularity and arm-angle drift picks the final solution.

IK and selection together take 3.5 ms on average within the 8 ms cycle.

### Delayed-spline interpolation

IK updates only when a new headset frame arrives, about every 16.7 ms, while servoJ needs a point every 8 ms. The solutions are stored with timestamps, and the output is the value from 50 ms in the past. Because of that lag there is real data on both sides of every output point, so the system **interpolates** with a Catmull–Rom spline instead of extrapolating. Extrapolation has to guess where the next target will be, and any jitter in the target makes the spline overshoot.

The only hard speed limit on this path is the per-step cap of 0.015 rad per 8 ms, or 107°/s. Normal tracking needs a median of 0.14° per step, so the cap rarely binds and only spreads a large jump over several steps.

## Orientation modes

### The height ceiling

Once the end-effector pose is fixed, the two wrist joints are fully determined; the arm angle changes only J1 and J3. Raising the end-effector while keeping the tool orientation fixed drives the wrist toward its ±47° limits:

| End-effector height | Wrist limit used |
| --- | --- |
| 451 mm | 79% |
| 491 mm | 96% |
| 511 mm | no solution |

This is a **495 mm ceiling**. Past it, IK fails, the arm stops while the hand keeps moving, and the next target is even further away. Tilting the tool slightly frees the wrist: at 491 mm a 20° tilt drops wrist usage from 96% to 61%, and a 30° tilt makes 591 mm reachable.

### Three modes

The hand's rotation is split into a **twist** around the tool axis, which costs no wrist travel, and a **tilt** of the tool axis, which is what buys height:

| Mode | Twist | Tilt | Height | Tool orientation |
| --- | --- | --- | --- | --- |
| `roll` | kept | dropped | capped at 495 mm | constant |
| `full` | kept | kept | 590 mm | follows the hand, can hit the wrist limit |
| `beside` | kept | scaled by wrist margin | 590 mm | follows the hand while it can, levels off when it cannot |

`beside` keeps a coefficient between 1 (`full`) and 0 (`roll`). When the last solution used more than 88% of the wrist range the coefficient slides down and gives back tilt; below 80% it climbs back and the tool levels itself. The hysteresis band between the two thresholds prevents chatter. The valve does not add height by itself. It keeps the arm from freezing when the operator turns the wrist in a direction that would exceed the wrist limits, which mostly happens when adjusting the tool angle at low and medium heights.

## Development

The videos below show how the system evolved, from queued Cartesian commands to the final servoJ pipeline.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="assets/video/dexhand_teleop/movel_single.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="assets/video/dexhand_teleop/movel_queued.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
</div>
<div class="caption">
    MoveL, one point at a time (left) and as a batched queue (right). Every point must finish before the next, so latency grows with hand speed.
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="assets/video/dexhand_teleop/local_ik_v1.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="assets/video/dexhand_teleop/servoj_raw.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
</div>
<div class="caption">
    First version with local IK and no smoothing (left), and servoJ without filtering, interpolation or planning (right). Without smoothing, headset jitter passes straight into the joint commands.
</div>

## System components

| Component | Role |
| --- | --- |
| `avp_servoj_teleop.py` | Main teleoperation loop |
| `ar5_poe_ik.py` | Closed-form product-of-exponentials IK |
| `ar5_ik_select.py` | Jump gate, configuration filter and cost ranking |
| `ar5_sew.py` | Configuration bits and arm-angle parameterization |
| `hand_retarget.py` | 25-joint hand skeleton to the hand's 6 actuators |
| `inspire_hand6.py` | Dexterous hand driver |
| `probe_servoj.py` | servoJ latency and stability measurement |
| `test_changes.py` | Offline regression tests before running on hardware |
