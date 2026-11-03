---
layout: page
title: Real-World Aligned Embodied Navigation
description: An object-goal navigation system that trains a CLIP-guided reinforcement learning agent on HM3D homes in NVIDIA Isaac Sim, with a ROS2 Humble mapping stack shared with the real Husky robot.
img: assets/img/autonav/isaac_hm3d_scene.jpg
importance: 2
category: Research
toc:
  sidebar: left
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/autonav/isaac_hm3d_scene.jpg" title="HM3D home in Isaac Sim" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A Husky robot in an HM3D home converted for NVIDIA Isaac Sim, rendered with semantic colours.
</div>

## Overview

Object-goal navigation asks a robot to find an object it has only been told about, such as "navigate to sofa", in a home it has never seen. Most agents for this task are trained in Habitat and stop at simulation. This project builds the full loop in a simulator with physically simulated robots and sensors, so that the perception, mapping and control stack the agent learns with is the same ROS2 stack that runs on a real robot.

The system was developed at [airvLab](https://smartlab.csc.liv.ac.uk/) between August and October 2023. It combines:

- **NVIDIA Isaac Sim 2022.2.1** for physics, rendering and the robot models (Clearpath Husky and Kobuki).
- **HM3D** indoor scans, converted from Habitat's `.glb` meshes into USD stages that Isaac Sim can load.
- **ROS2 Humble** for sensor streaming, SLAM and the robot's transforms.
- **CLIP** and **PPO** (Stable-Baselines3) for a language-conditioned navigation policy.

## System architecture

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/autonav/system_overview.png" title="System architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The four components of the system. DATA supplies environments and episodes, SIMULATION runs the Isaac Sim world, and BRAIN observes the world through the ROS2 bridge and sends back control actions.
</div>

- **DATA** loads an HM3D environment and an ObjectNav episode: the start pose, the target category and every instance of that category in the house. Habitat's Y-up coordinates are converted to Isaac Sim's Z-up frame on the way in. The training split has 143 houses and the evaluation split 36.
- **SIMULATION** places the house and the robot on the stage, steps the physics and drives the robot with a differential-drive wheel-base pose controller.
- **BRAIN** holds the ROS2 nodes for perception, localization and mapping, together with the reinforcement learning policy that decides where to go next.
- **UTILS** provides shared tools such as argument parsing and console messages.

## Simulation to ROS2

An OmniGraph action graph inside Isaac Sim publishes the robot's sensors to ROS2 on every simulation tick:

| Sensor | Isaac Sim source | ROS2 topic |
| --- | --- | --- |
| RGB camera | Camera on `base_link`, through a viewport render product | `robot_view` |
| 2D LiDAR | Range sensor on `base_link` | `scan` (`sensor_msgs/LaserScan`) |

A gmapping SLAM node turns the laser scans into an occupancy grid on `map`. The project extends a ROS2 port of `slam_gmapping` with a **map-clear service**, so that the training loop can wipe the map at the start of every episode instead of restarting the node. The official `slam_toolbox` failed with a `libglog` symbol conflict when launched from the Isaac Sim process, which is why the pipeline uses this gmapping variant.

Inside the training process, a ROS2 subscriber converts each occupancy grid into a grayscale image (free, occupied and unknown cells) and draws the agent's position on it from the `tf` transforms. This image is the map the policy sees.

## Navigation agent

### Observations

At each step the agent receives four inputs:

| Input | Shape | Content |
| --- | --- | --- |
| `img` | 124 × 124 × 3 | RGB view from the robot camera |
| `map` | 124 × 124 × 1 | Occupancy map with the agent drawn on it |
| `text_emb` | 512 | CLIP ViT-B/32 embedding of "navigate to ⟨target⟩" |
| `img_emb` | 512 | CLIP ViT-B/32 embedding of the current view |

### Feature extractor

A custom Stable-Baselines3 feature extractor encodes each input separately and concatenates the results into one 2432-dimensional feature:

- The RGB view goes through a frozen, ImageNet-pretrained **ResNet-50** (2048-d).
- The map goes through a 5 × 5 convolution and a linear layer (128-d).
- The CLIP text and image embeddings each go through a linear layer (128-d each).

### Actions

The policy outputs three numbers $$(x, y, c)$$. The pair $$(x, y)$$ is a target cell on the occupancy map, which is converted into a goal in the simulator frame using the agent's map position and the map resolution, then tracked by the wheel-base controller. The third value $$c \in [0, 1]$$ is a termination confidence: at each step the agent declares that it has found the target with probability $$c$$. An episode ends when the agent declares termination or after 500 steps.

### Reward

The reward combines a sparse outcome term with dense guidance from CLIP:

$$
r_t = 25 \cdot \mathbb{1}[\text{success}] \;-\; 25 \cdot \mathbb{1}[\text{false stop}] \;+\; \frac{1}{10}\, s_{\mathrm{CLIP}}(o_t, \text{target})
$$

A declared stop counts as a success when the robot is within 1 m of any instance of the target category. The CLIP term $$s_{\mathrm{CLIP}}$$ is the softmax similarity between the current view and the prompt "navigate to ⟨target⟩" against the prompts for all six categories, so the agent is rewarded for bringing the target into view before it reaches it. The targets are TV monitor, sofa, chair, plant, toilet and bed.

### Training

The policy is trained with PPO using a multi-input policy:

| Setting | Value |
| --- | --- |
| Rollout length | 250 steps |
| Batch size | 50 |
| Epochs per update | 20 |
| Learning rate | 1 × 10⁻³ |
| Discount γ | 0.99 |
| Value loss coefficient | 0.5 |
| Gradient clipping | 1.0 |
| Total timesteps | 500,000 |

Each episode loads a random training house and episode, places the robot at the start pose, clears the SLAM map and starts a fresh trajectory. Training runs headless with Isaac Sim livestreaming for monitoring, and the ROS2 mapping pipeline is launched from the training process itself.

## Engineering

- **Containerized stack.** A Docker image built on NVIDIA's Isaac Sim 2022.2.1 container compiles ROS2 Humble from source and installs CLIP, Stable-Baselines3 and OpenCV into Isaac Sim's Python, so simulation, ROS2 and learning run in one environment.
- **Two robots.** The task supports both the Clearpath Husky and the Kobuki as USD assets, selected by a command-line flag.
- **Launch files.** ROS2 launch files bring up the static transforms, SLAM, Nav2 and the map processor in one command for development outside training.

## Status

The infrastructure is complete: the brain components were finished in September 2023 and parallel headless training in October 2023. Training of the baseline agent and its evaluation against map-based ObjectNav methods such as Active Neural SLAM, SemExp and VLMaps were the next steps.
