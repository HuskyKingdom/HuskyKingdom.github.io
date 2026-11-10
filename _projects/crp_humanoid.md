---
layout: page
title: "crp Humanoid: From Motion Tracking to Steady Walking"
description: Taking our in-house humanoid crp from dancing in simulation to walking on a gamepad and playing motions at the press of a button, with world-model policy learning from SuperMimic.
img: assets/img/crp/cover.jpg
importance: -2
category: Industrial
toc:
  sidebar: left
---

This project covers how we took our own humanoid from "dances in simulation" to "walks on the floor with a gamepad and plays motions at the press of a button": how we trained motion tracking, how the walking policy evolved, and what finally runs on the robot.

<div class="row justify-content-center">
    <div class="col-sm-5 mt-3 mt-md-0">
        {% include video.html path="assets/video/crp/walk_outdoor.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
</div>
<div class="caption">
    crp walking outdoors, driven by the gamepad.
</div>

## The robot and the method

**The robot.** crp (model RHM-D1F00-B):

- 59.7 kg, with legs about 1.3× as long as a Unitree G1's;
- 31 motors, 23 of them driven by the policy (6 per leg, waist yaw, 5 per arm); waist roll/pitch, head and wrists are held still;
- parallel-linkage ankles (two motors per ankle);
- ROS2 underneath.

**The method.** Training uses SuperMimic from [MorphMimic](https://github.com/HuskyKingdom/morphmimic):

- A **world model** learns "state + action → next state" from simulation data.
- The **policy** "imagines" 32 steps inside the world model. Rewards are differentiable with respect to the states, so the policy is updated by backpropagating straight through the imagined rollout. This is not trial-and-error sampling as in PPO, and it needs far fewer samples.
- The world model only describes physics and knows nothing about the task. Once trained for motion tracking, it is reused for walking, and only a new policy is trained.

Training ran on an offline H100. The policies run on the robot's own x86 computer.

## Step 1: Motion tracking

**Data.** We transferred the BONES-SEED motion data from the G1 to crp joint by joint: 127k clips, 258 hours in total.

**Training.** One H100, 40,000 iterations, about two days. In simulation:

- all 16 test motions completed;
- mean joint error 0.090 rad;
- push recovery 48/48.

**What broke on the real robot:**

| Symptom | Cause | Fix |
|---|---|---|
| The robot moves on its own while "standing" | The ankles are linkages, but we sent joint angles straight to the linkage motors | Ported the vendor's parallel-ankle solver |
| A whole leg goes limp mid-motion | A CAN bus drops out; frozen readings keep being published | Any missing or frozen motor damps the whole body immediately |
| Actions often arrive one tick late | Inference takes ~9 ms, but we only waited 10 ms | Wait 18 ms |
| Unplugging the laptop cable stops the robot | ROS2 traffic was routed through that cable | Route the robot's internal traffic over loopback |

In the end, three gamepad buttons trigger three motions: a wave, pointing forward, and a short disco dance.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="assets/video/crp/tracking_1.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="assets/video/crp/tracking_2.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="assets/video/crp/tracking_3.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
</div>
<div class="caption">
    Motion tracking on the real crp.
</div>

## Step 2: Walking

Walking is not a separate RL setup. It is a new "velocity command" task inside the same framework:

- the goal is [forward speed, sideways speed, turn rate] plus a gait clock that tells each foot when to lift;
- the rewards are computed on world-model states.

### Part one: from walking at all to walking decently

Every version followed the same loop: the robot shows a problem → recorded data tells us why → we reproduce it in simulation → we fix it.

1. **Tripping over its own feet.** Foot lift was measured at the ankle joint: the ankle rose 9 cm while the toe cleared the floor by only 0.5 cm. We now measure the lowest point of the sole.
2. **Stomping.** We added a double-support phase, a smooth swing curve, flat feet, a minimum foot spacing and counter-swinging arms. The result, `natural`, was the first version that walked well on the robot.
3. **Leaning left until it fell.** The real ankle roll was only about 1/5 as stiff as in training: the linkage motors' gains were tuned for pitch, and the roll feedforward had been switched off. Adding the feedforward every control tick mostly fixed the lean.
4. **Left knee at 82 °C, then off the bus.** The policy had learned to stand braced on one leg, with the left knee holding 60 Nm. A joint-torque penalty plus a left-right symmetry loss brought each knee to about 29 Nm.
5. **Swaying upper body, tipping over when stopping.** The motors holding the waist were only half as stiff as commanded, with torque capped at about 42 Nm. We modelled the waist as a spring in simulation and trained a wider stance.
6. **Leaning back and stepping in place.** Both ankle pitch zeros were off by about 3°, so the real robot's center of mass (CoM) sat 3–4 cm further back than in simulation. We randomized ankle zero errors in training and kept enough CoM margin.
7. **Adversarial motion priors (AMP)**, to look more human: the feet dragged and turning drifted, so we dropped it.

<div class="row">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include video.html path="assets/video/crp/sim_smoke_200.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include video.html path="assets/video/crp/sim_gait_dp_2000.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
    <div class="col-sm-6 mt-3">
        {% include video.html path="assets/video/crp/sim_gait_dp_4000.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
    <div class="col-sm-6 mt-3">
        {% include video.html path="assets/video/crp/sim_natural_8000.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
</div>
<div class="caption">
    Walking policies in simulation under the same velocity-command sequence: a 200-iteration smoke test (top left), the gait version at 2,000 and 4,000 iterations (top right, bottom left), and <code>natural</code> at 8,000 iterations (bottom right).
</div>

<div class="row justify-content-center">
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include video.html path="assets/video/crp/walk_lab_early.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include video.html path="assets/video/crp/walk_lab.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
</div>
<div class="caption">
    Walking tests on the real robot in the lab: an earlier version (left) and a later one (right).
</div>

### Part two: human-like and stable

**Motion capture as the yardstick.** One forward-kinematics pipeline computes the same quantities from pelvis attitude and joint angles: stance-knee angle, how far the CoM is ahead of the ankle, and pelvis pitch. That put human mocap, simulation and the real robot side by side. The policy's legs were too bent (stance knee 23–27°, humans 7–10°), and on the real robot the CoM sat behind the ankles.

Then came a game of whack-a-mole:

- **Straight-leg posture reference + CoM term:** simulation improved, but the real robot leaned back. The policy didn't trust its IMU, because encoder zero errors never made it into imagination.
- **Encoder zero errors in imagination:** the policy started using the IMU, but now swung its arms sideways for balance.
- **A separate arm constraint:** the arms calmed down, but the ankle motors overheated (heat proxy 303).
- **Ankle torque penalty:** the policy learned to crouch against the ankles' mechanical stops, since the stops carry load for free.
- **Joint-limit margin + stance-leg posture cost:** no more crouching, but the CoM slowly drifted back again.

**What worked: using the human gait itself as the reference.** We extracted each joint's mean curve over the gait cycle from mocap, scaled it with speed, and had the policy track it. Sideways arm swing dropped from 30° to 5°, matching humans. This version (`human7`, checkpoint 6000) stopped and stood steadily on the real robot, but still leaned and drifted left while walking.

### The last round: removing the lean and the drift

**Finding the cause (all from real-robot data):**

- Standing with both soles flat, the joint readings put the right foot at a 2.4° tilt: the right ankle roll has a zero error.
- The right ankle roll's tracking error is 35–50% larger than the left's, and the right knee and hip yaw deliver 10–15% less torque: **the right leg is softer**.
- In simulation, a right leg at 80% stiffness reproduces the real left lean and drift. A zero error alone does not.

**Detours along the way:**

- **A pelvis-tilt penalty** looked great in simulation, but the real robot shook at 6–7 Hz. The policy had become a high-gain controller, and the robot's tens of milliseconds of latency made it oscillate. A 40 ms action delay in simulation reproduces it, and every version is now tested against it.
- **Tighter posture constraints** sent the arms flailing again, so we added an arm constraint that never saturates.
- **The lean's direction depended on the random seed.** A symmetry loss only encourages symmetry; the policy itself was never exactly symmetric.

**The final recipe**, informed by the randomization used in mainstream frameworks such as [Booster Gym](https://arxiv.org/pdf/2506.15132) and by work on [joint-offset calibration](https://arxiv.org/abs/2609.02306) and [symmetric policies](https://arxiv.org/pdf/2508.01247):

| Problem | What we did |
|---|---|
| The two legs differ | In imagination, each leg joint gets a random stiffness of ±25%, independently per side; zero errors affect both the reading and the executed target |
| Control latency | 0–20 ms of random latency in imagination |
| Lean direction depends on the seed | **Symmetric inference**: average the policy's action with the mirrored action on the mirrored input. The policy becomes exactly left-right symmetric, with no retraining |
| Doesn't walk straight | Gamepad **heading hold**: when only pushing forward or back, the IMU locks the heading |
| Hot ankles | The full ankle-motor torque penalty, tripled |

**Acceptance.** One night, six GPUs, several variants in parallel. Every checkpoint was automatically checked against the failures we had already hit:

- falls and speed tracking;
- shaking under a 40 ms delay;
- flailing arms;
- ankle load;
- lean and drift during a 20 s straight walk, with normal legs and with a "soft right leg";
- stopping cleanly.

The winner was `p_4000`. Under conditions closest to the real robot (right leg 20% softer, 20 ms latency, right ankle zero error):

| | Pelvis sideways offset | Roll |
|---|---|---|
| Previous best, human7_6000 | 1.8 cm | 1.42° |
| Final, p_4000 | 0.6 cm | 0.28° |

On the real robot it is the best-walking version so far.

## How it runs on the robot

Everything runs on the robot's own computer; no laptop needed:

```
gamepad ──► joy_node ──► crp_node (50 Hz) ◄──► motion_control (vendor ROS2 motor driver)
                           │  · sticks → velocity command (with heading hold)
                           │  · parallel-ankle solver + roll feedforward, IMU mounting correction
                           │  · motor dropout / frozen readings → damp the whole body
                           ▼
                        policy process (walk_clips)
                           · walking policy (ONNX on CPU, ~1.6 ms per tick with symmetrization)
                           · motion-tracking policy (wave / point / dance)
```

- **Starts at boot.** A systemd service starts the stack in a damped state. Start stands the robot up, A starts walking, and B damps at any time.
- **Gamepad.**
  - Left stick: forward/back and sideways. Right stick: turn.
  - X, Y and RB play motion clips: the robot stops, takes a settling step, plays the clip, then resumes walking.
- **Configuration.** Everything lives in one `walk.env`: which policy, symmetric inference on/off, heading-hold gain, ankle feedforward, IMU correction.
- **Data.** Every run records joints, IMU and motor torques for analyzing the next version.
- **Updates.** One command pushes a new policy and code to the robot. Files are written to a temp name and then renamed, so the running service never reads a half-written file. After editing `walk.env`, restart while damped.

## Takeaways

- **Build the yardstick before tuning rewards.** Comparing human mocap, simulation and the real robot with the same metrics is what made problems clear.
- **Reproduce real-robot problems in simulation first.** The left lean, the backward lean and the shaking were all reproduced in simulation before we found their real causes.
- **Every patch invites the next shortcut.** Rather than piling on penalties, give the policy a good reference (human gait) and back it with a strict acceptance checklist.
- **Add latency, actuator and zero-offset randomization early.** Mainstream open-source frameworks include them from the start; we had to be forced there by the robot, one failure at a time.
- **Guarantee symmetry by construction.** A loss can only encourage it; symmetric inference guarantees it.

Next up: having the vendor recalibrate the parallel-ankle zeros, and voice control ("Xiaonuo, Xiaonuo, wave!").
