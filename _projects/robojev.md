---
layout: page
title: RoboJev
description: Testing the new generation of decision models (Jev, Laya, PlayJev and the OpenAI Decisions API) as the brain of a robot arm for the atomic action of grasping.
img: assets/img/robojev/cover.jpg
importance: -1
category: Industrial
toc:
  sidebar: left
---

<div class="row justify-content-center">
    <div class="col-sm-5 mt-3 mt-md-0">
        {% include video.html path="assets/video/robojev/robotjev_demo.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
</div>
<div class="caption">
    RobotJev on a Rokae AR5 arm: open-vocabulary grasping from a language instruction or a click, with grasp force adapted to the object.
</div>

## Overview

Since September 2026 a new kind of model has appeared: the **decision model**. It does not generate text. The caller gives it the current state and a question with fixed options, and one forward pass returns a probability for every option. Questions come in three forms: multiple choice, yes/no predicates and scores.

This makes decision models a natural fit for closed-loop robot control, where a decision is needed at every step. A generative model call took 4–5 s in our tests, while a decision model answers in one pass, in tens of milliseconds when run locally. Its output is already a distribution over actions: the options are moves such as up, down, left, right and close the gripper, the model picks one, and code executes it with step sizes and safety checks kept in rules.

RoboJev tests whether these models can drive a 7-DoF Rokae AR5 arm through the atomic action of **grasping**, in two projects:

- **[RobotJev](https://github.com/Vector-xxr/RobotJev)** (September 23–30, 2026) uses the open models Laya and PlayJev with a two-finger gripper.
- **[RobotDecision](https://github.com/Vector-xxr/RobotDecision)** (October 12–15, 2026) uses the OpenAI Decisions API with an Inspire dexterous hand.

## RobotJev

### From Jev to Laya to PlayJev

**Jev**, released by TypeSafe AI on September 15, 2026, is only available as a hosted API, with no weights to run locally or fine-tune. Its open alternative **Laya** (Convai Innovations, Apache 2.0) is a 421M-parameter ModernBERT-large encoder with a decision head. It is non-autoregressive: the input goes through the encoder once and the option probabilities are read from the head.

Laya only reads text, so the scene first went through YOLO detection and the offset was written as a sentence, such as "the apple is 18 centimetres to the left of the gripper". Phrasing decided everything. Written as `dx=+0.180 m`, the model got 1 of 8 directions right; rewritten in plain words and with the magnitude removed from the question, it reached **87.5%** on six-way direction choice (chance is 16.7%). It could not read magnitudes at all: as the distance went from 30 cm down to 1 cm, its probability of choosing a large step stayed between 0.11 and 0.23, and its probability of closing the gripper did not change with distance.

**PlayJev** (OmniJev) can see. It is a Jev-style multimodal model built on Qwen3.5-0.8B and trained on frames from small games: given a frame and action options, one forward pass returns a probability for each action. Fed real camera images, it failed. Mirroring the image did not swap its answers, and painting out the target made "left" more likely.

### Turning the robot into Snake

The fix was to show PlayJev a world it already knows. The offset between gripper and target is measured by the cameras and calibration, then drawn as a **Snake board**: a dark blue grid with a white block for the gripper and a red block for the target, asked with the exact wording of the Snake game. On this board PlayJev chose the right direction in **120 of 120** random states, mirrored boards included, and made 250 correct decisions in a simulated closed loop of 8 targets.

Two details mattered:

- Only axes with at least 30% of the largest offset are drawn. With both axes drawn, the model often corrected the smaller one first.
- Oscillation in the loop came from step sizes, not from the model, which chose correctly at every step. The per-axis dead zone was set to half a fine step, and the coarse or fine step is chosen from the error on the selected axis rather than the total distance.

### Grasping pipeline

PlayJev only aligns the last few centimetres. Everything else is measured: where the object is, how tall and wide it is, and how hard to squeeze.

1. **Top-down localization.** A Gemini 335 depth camera above the table runs YOLO-World on a few phrasings of the object from the instruction, with a CLIP check against background descriptions. MobileSAM segments the object, and depth gives its point cloud, support plane, height, width and grasp point. Tall objects are grasped at the top section, so a bottle is grasped by its cap. In click mode, the user clicks the object instead.
2. **Graspability.** There is no workspace box. An object is graspable only if the width above the grasp height fits the 98 mm gripper with margin, and inverse kinematics has a solution at the grasp, pre-grasp and transit points.
3. **Approach.** The arm moves to a pre-grasp point 10 cm back along the wrist camera's line of sight, then descends vertically.
4. **Wrist hand-off.** The arm jogs ±1 cm along each axis, and optical flow with depth gives the mapping from arm motion to wrist-image motion. If the wrist and top cameras disagree on the object's height, as with transparent objects, the arm falls back to a straight approach from the top-camera estimate.
5. **Force and grasp region.** CLIP classifies the object as fragile, soft or hard, which sets grip force and closing speed, and picks the upper, middle or lower part to grasp.
6. **PlayJev servoing.** At each step the offset between the grasp point and the gripper centre is drawn as a Snake board and PlayJev picks a direction, which is mapped to arm motion. Steps are 12 mm when far and 5 mm when close, and alignment is reached below 6 mm.
7. **Grasp and verify.** The gripper closes at the chosen force and lifts 60 mm. Success requires both the gripper reading to match the object width and the object to move with the wrist camera.

The arm stops if it receives no command for 0.25 s, so a watchdog thread resends targets while perception runs YOLO-World, CLIP and SAM.

### Results

| Test | Result |
| --- | --- |
| Laya, direction (6-way) | 87.5% (21/24), chance 16.7% |
| Laya, step size (large or small) | 0%, chance 50% |
| PlayJev on real camera images (zero-shot) | Unusable |
| PlayJev on Snake boards | 120/120 directions, unchanged under mirroring |
| Real-robot grasping | Ping-pong ball (2 mm alignment error), blue socket (3.7 mm), yellow tape (4.7 mm), egg (fragile setting), water bottle (by the cap), ball in a tray (straight approach). Failed on a white charger. |

## RobotDecision

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.html path="assets/video/robojev/robotdecision_run.mp4" class="img-fluid rounded z-depth-1" controls=true muted=true %}
    </div>
</div>
<div class="caption">
    A RobotDecision run: top-down view with lettered arrows, wrist view with the green target ring, and side view. Each step shows the chosen action and the stage it belongs to.
</div>

RobotJev needs a heavy vision pipeline so that the model only chooses a direction. RobotDecision removes it: the **OpenAI Decisions API** looks directly at three camera views (top-down, wrist and side) and picks one preset action per step, driving the AR5 with an Inspire RH56E2 hand to grasp a socket on a whiteboard. The hand uses only thumb and index finger as a two-finger gripper.

### What the API is like

- It only chooses among given options and never outputs coordinates or distances, so step sizes and safety checks stay in rules.
- Latency is about 0.4 s for text and 0.7 s with a wrist image.
- Repeated identical requests return identical answers, so uncertainty cannot be estimated by resampling.
- Multiple-choice answers depend on option order and are overconfident. Every choice is therefore asked twice, in forward and reversed order, and executed only if both agree. Yes/no probabilities are more reliable and are used for gating.
- A fixed scene description must be part of the input. Without one, it did not recognize robot fingers filling a quarter of the frame; with a sentence on where the camera is mounted and what the fingers look like, it identified them every time.

### Finding the right questions

The questions were designed offline on recorded images before moving the arm: 200 sets of three views from a replayed VR demonstration, labelled by forward kinematics, plus 50 sets on a grid above the socket.

| Version | Question | Accuracy |
| --- | --- | --- |
| v1 | Two images, choose 1 of 10 actions | 21%, essentially guessing |
| v2 | Lettered direction arrows drawn on the top view, letters shuffled every frame | 83% horizontal, 83% mirrored |
| v3 | Arrows, plus a green ring on the wrist view and two yes/no questions | 85% overall; 94% beyond 150 mm, 12% within 60 mm |
| v4 | Adding the side view | 53% |
| v5 | One combined "next step" question with arrows on the wrist view | 35%, closes too early |
| Split | Wrist view only: "which arrow points from the ring to the socket"; a dashed inner ring for closing | 77% direction; 8/10 correct closes, 0/72 false closes |

Up close, the top view is blocked by the hand, and the model cannot convert a wrist-view offset into a top-view arrow, because the wrist camera is rotated 45° and mirrored relative to the top view. The final design therefore asks by distance: the top view when far, a dedicated wrist question when near, and the double ring to decide when to close.

### Final controller

Each step first asks where the socket is relative to the wrist ring, which selects the stage:

| Stage | Images given | Options | Step |
| --- | --- | --- | --- |
| Coarse (socket not in wrist view) | Top view with four lettered arrows | Arrow | 20 mm |
| Approach (outside the ring) | Top and wrist views with matching arrows | Arrow | 15 mm |
| Inside the ring | Wrist view (ring, centre, arrows) and side view | Arrow, descend or close | 5 mm move, 8 mm descent |

Grasp success is read from the fingers: if both fingers are blocked before reaching their commanded position, the socket is held; on an empty grasp the hand opens, lowers 5 mm and tries again, up to 4 times. Safety uses only physical criteria: reachable IK with under 1 mm error, joint-limit margins, at most 10° per joint per step and a minimum flange height.

### Results

All **6 real-robot runs held the socket on the first close**, taking 57–107 steps and 176–262 s including placing it back and returning home. The acceptance run took 102 steps and 185 API calls in 262 s, with 35 steps re-asked because the forward and reversed answers disagreed.

Two limits showed up:

- **Speed.** A call takes 0.4–0.7 s (median 0.57 s). That is faster than a 4–5 s generative call but about 25 times slower than local PlayJev at 23 ms, and every choice is asked twice.
- **Rotation.** Asked how to rotate the wrist, the model was right 13 of 28 times across both orders (46%), no better than chance, so the grasp orientation is copied from a VR demonstration.

## Conclusions

Decision models bring two real advantages to vision-language-action tasks:

- **Broad multimodal knowledge without training.** They understand spatial language (Laya, 87.5% on direction), tell whether a target is in view (OpenAI, 91–100%) and read arrows drawn on an image (83%, unchanged under mirroring).
- **Low latency.** One forward pass gives action probabilities: 23 ms for local PlayJev and 0.4–0.7 s for the OpenAI Decisions API, against 4–5 s for a generative call.

Embodied use still needs **domain adaptation**. Zero-shot, the models cannot read magnitudes, cannot see geometry in real images and cannot handle viewpoint rotation. In both projects the adaptation lives entirely in the input, by drawing the state as a game board, adding arrows and rings to images and splitting questions, while the models themselves are not trained.

## Hardware

| Component | RobotJev | RobotDecision |
| --- | --- | --- |
| Arm | Rokae AR5 (7-DoF) | Rokae AR5 (7-DoF) |
| End effector | Hangkai HKVR-TG9801 two-finger gripper (98 mm) | Inspire RH56E2 dexterous hand |
| Top camera | Orbbec Gemini 335 | Orbbec Gemini 335 |
| Wrist camera | Orbbec Gemini 305 | Orbbec Gemini 305 |
| Side camera | — | UGREEN 1080p |
| Compute | RTX 5070 Ti 16 GB (PlayJev uses about 3.6 GB) | OpenAI Decisions API |

## Resources

- [RobotJev source code](https://github.com/Vector-xxr/RobotJev)
- [RobotDecision source code](https://github.com/Vector-xxr/RobotDecision)
- [PlayJev model card](https://huggingface.co/omnijev/playjev-0.8b)
- [Jev, Laya and the emerging role of decision models](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- [OpenAI Decisions API public beta](https://letsdatascience.com/news/openai-opens-decisions-api-public-beta-b69a1e40)
