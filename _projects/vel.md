---
layout: page
title: "VEL: Vision-Grounded Energy Critics"
description: Vision-grounded energy critics for physically feasible vision-language-action policies. A compact energy head corrects each action of a frozen VLA with one gradient step, with no rollouts or retraining.
img: assets/img/vel/energy_correction.jpg
importance: 0
category: Research
toc:
  sidebar: left
---

**Vision-Grounded Energy Critics for Physically Feasible Vision-and-Language Action Policies.** Under review.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/vel/energy_correction.jpg" title="Energy-based action correction" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Energy-based action correction. The energy head scores each state–action pair, and one local descent step toward lower energy corrects the action so the bowl no longer blocks the cabinet.
</div>

## Overview

Vision-and-language action (VLA) models trained by behavior cloning map images and an instruction to robot actions, but they have no internal signal for whether an action fits the scene. A grasp that is slightly off, or a bowl placed one centimeter too close to a door, is executed as confidently as a good action. In long-horizon tasks such a small error can break every step that follows.

**Vision-Dominated Energy Learning (VEL)** adds a compact energy head to any frozen or fine-tuned VLA. The head maps the VLA's hidden representation and a candidate action to a scalar energy, where lower energy means the action fits the scene better. It is trained with a contrastive objective on the demonstration data alone. At inference, one gradient step moves the proposed action toward lower energy. VEL needs no environment rollouts, no extra data and no change to the VLA backbone.

- On LIBERO-Long, VEL raises **OpenVLA-OFT from 94.4% to 95.8%** and **π0.5 from 92.4% to 93.6%**, removing 24% and 16% of the remaining failures.
- On a real screw-insertion task with a π0.5 backbone, full-task success rises **from 35% to 50%**, and the gain comes from the precision-demanding insertion stage.
- The head adds fewer than **4.0M parameters** and **4.61 GFLOPs** per correction step, 0.4% over OpenVLA-OFT and 5.2% over π0.5.

## Method

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/vel/framework.jpg" title="VEL framework" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Overview of VEL. The frozen backbone yields hidden features and a behavior-cloned action. The action embedding queries the state tokens through multi-head attention, a SiLU feed-forward readout gives a scalar energy, and its gradient corrects the action in one residual step.
</div>

### Why vision tokens

In representative VLA architectures, vision tokens make up over 85% of the last-layer hidden sequence and receive over 65% of the attention mass in the final decoder layer. This visual stream already encodes object poses, contact geometry and spatial relations, which is exactly the information that separates a feasible action from a subtly flawed one. VEL reads its energy from this hidden sequence.

### Energy head

Two projectors map the hidden features $$h_t^{(-1)}$$ to state tokens $$Z_h$$ and the candidate action $$a_t$$ to a single embedding $$z_a$$. The action embedding then attends to the state tokens:

$$
Z_h = P_h\big(h_t^{(-1)}\big), \quad z_a = P_a(a_t), \quad \tilde{z}_a = \mathrm{MHA}(Q = z_a,\ K = Z_h,\ V = Z_h)
$$

Using the action as the query lets it attend to the task-relevant regions of the scene. A SiLU feed-forward block and a linear readout turn $$\tilde{z}_a$$ into a scalar energy. The tokens from which the backbone decodes its own action are masked, so the head judges the action against the scene and not against the backbone's prediction.

### Contrastive training

The head is trained with in-batch InfoNCE. For each state, its expert action is the positive and the actions of the other states in the batch are the negatives:

$$
\mathcal{L}_{\mathrm{VEL}} = \frac{1}{B} \sum_{b=1}^{B} -\log \frac{\exp\big(-E_w(h_b, a_b^{+})/\tau\big)}{\sum_{j=1}^{B} \exp\big(-E_w(h_b, a_j)/\tau\big)}
$$

Gradients stop before the backbone, so only the head's weights $$w$$ are updated. At the population optimum the objective recovers the log density ratio between the conditional expert action distribution and the action marginal. Its action gradient is therefore the expert score minus the marginal score, which measures how well an action fits this particular state rather than how common it is overall.

### One-step residual correction

At inference VEL takes a single gradient step from the behavior-cloned action and clips the result to the action bounds:

$$
\tilde{a} = a_{BC} - \alpha\, \nabla_a E_w\big(h_t^{(-1)}, a_{BC}\big), \qquad a_{out} = \mathrm{clip}(\tilde{a}, -1, 1)
$$

Under local smoothness, this step is guaranteed to lower the learned energy. Only one step is taken on purpose: the energy is calibrated near the demonstrations, and further descent or search over it lowers the energy without improving the action. For action-chunking VLAs such as OpenVLA-OFT, each action in the chunk is corrected independently. For diffusion VLAs such as π0.5, the correction is applied to the final denoised action.

## Results

### LIBERO

Success rates (%) on LIBERO. VEL rows are mean ± std over five energy-head training seeds.

| Method | Spatial | Object | Goal | Long | Average |
| --- | --- | --- | --- | --- | --- |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.8 |
| π0.5 + VEL | 98.2±0.4 | 97.6±0.8 | 98.3±0.5 | **93.6±0.6** | **97.0±0.2** |
| OpenVLA-OFT | 97.4 | 98.4 | 96.8 | 94.4 | 96.7 |
| OpenVLA-OFT + VEL | 97.9±0.9 | 98.2±1.7 | 97.5±0.8 | **95.8±0.9** | **97.3±0.3** |

The LIBERO-Long gains are significant for both backbones (p = 0.030 for OpenVLA-OFT, p = 0.011 for π0.5). The three shorter suites already run between 96.8% and 98.8% and show smaller changes of mixed sign.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/vel/libero_rollout.jpg" title="LIBERO-Long rollout" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    LIBERO-Long task 8. Top: OpenVLA-OFT with VEL succeeds. Bottom: OpenVLA-OFT alone fails. VEL adjusts the end-effector during the initial grasp, so the mug clears the microwave door instead of catching on it. Blue/green: closed/open gripper; yellow crosses: end effector.
</div>

### What makes the gain

The gain could come from any test-time refinement, from the extra parameters, or from energy learning in general. Controlled comparisons on OpenVLA-OFT with the same backbone, data and step size rule these out:

| Method | Long | Average |
| --- | --- | --- |
| OpenVLA-OFT (base) | 94.4 | 96.7 |
| + Best-of-N over the energy (noise injection) | 92.6±3.4 | 95.4±0.4 |
| + IBC-style CEM search over the energy | 91.8±2.8 | 94.5±0.7 |
| + Direct correction head, matched capacity | 69.6±0.5 | 84.9±0.4 |
| + Score-matching energy head | 92.8±1.0 | 96.7±0.5 |
| + VEL | **95.8±0.9** | **97.3±0.3** |

- **Search over the same energy falls below the base policy.** Replacing the base action discards the kinematics it established, and the deviation compounds over the episode.
- **Extra capacity alone does not help.** A regression head of the same size has to average over multimodal expert deviations and drops to 69.6% on LIBERO-Long.
- **The contrastive objective matters.** A score-matching head keeps a state-independent pull toward globally frequent actions and falls below the base on LIBERO-Long, while the contrastive field subtracts that term.

### Real robot

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/vel/real_robot.jpg" title="Real-robot screw insertion" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Real-robot screw insertion with a π0.5 backbone: (a) the setup, (b) π0.5 approaches off-axis and fails, while with VEL it inserts, (c) success over 100 runs per policy.
</div>

The setup is a Rokae xMate arm with an Inspire RH56 dexterous hand and two RGB cameras. The robot picks up a 3D-printed screw and inserts it into a base, a step that tolerates little positional error. π0.5 is fine-tuned on 300 teleoperated demonstrations, and the energy head is trained on the same demonstrations with the backbone frozen. Grasping is almost unchanged (82% vs 83%), while insertion rises from 44% to 60% and the full task from 35% to 50%. The episodes VEL recovers are those that grasped correctly and would have failed on the precise placement.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/vel/real_robot_energy.jpg" title="Energy during real-robot execution" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Energy during real-robot execution for the grasp (blue) and insertion (red) stages, mean and one standard deviation over ten episodes per policy. With VEL the band is narrower and the mean energy lower in both stages, with the largest reduction in insertion.
</div>

### The learned energy

<div class="row justify-content-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.html path="assets/img/vel/energy_landscape.jpg" title="Learned energy landscape" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The learned energy landscape around expert actions; the red dot marks the expert action.
</div>

The energy surface forms a clear valley with the expert action at its minimum, and its magnitude grows with the ℓ1 distance to the expert action. Episode-level mean energy separates successful from failed LIBERO-Long episodes with an AUROC of 0.749. Under object perturbation on LIBERO-Spatial, VEL improves success by 3.4 points; under camera perturbation it gives no gain.

### Inference cost

| Component | Latency (ms) | GFLOPs |
| --- | --- | --- |
| OpenVLA-OFT | 103.3 | 1178.83 |
| + VEL | 106.4 | 1183.44 |
| π0.5 (co-located) | 80.2 | 88.92 |
| + VEL | 83.3 | 93.53 |

The energy head's forward pass takes 0.7 ms and its gradient 2.4 ms, so one correction step adds about 3 ms on top of the backbone.

## Conclusion

VEL is a backbone-agnostic energy critic that refines behavior-cloned actions with a single gradient step at inference. With under 4.0M parameters and a small fraction of the backbone's compute, it removes about a quarter of the remaining long-horizon failures on LIBERO and raises real-robot screw insertion from 35% to 50%, which makes it a practical add-on for existing VLA pipelines.
