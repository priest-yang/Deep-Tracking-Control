# Deep Tracking Control with Lite3


### Main contribution

To summarize, this project aims at combining the traditional MPC-based and terrain-aware foothold planner with the deep reinforcement learning(DRL) . The goal is to achieve robust control in extremely risky terrains such as [stepping stone](legged_gym/utils/terrain.py).

You can find the modifications in [`legged_robot_dtc.py`](legged_gym/envs/base/legged_robot_dtc.py) and [`legged_robot_config.py`](legged_gym/envs/base/legged_robot_config.py). 



### Foothold Planner

In this project, we adapt a method similar to [TAMOLS](https://arxiv.org/abs/2206.14049) and [Mini-Cheetah](https://arxiv.org/abs/1909.06586). 

An estimated foothold will firstly be calculated by the formula:

$$
r_i^{cmd} = p_{shoulder, i} + p_{symmetry} + p_{centrifugal}
$$


where

$$
p_{shoulder,i} = p_k + R_z(\Psi_k)l_i
$$

$$
p_{symmetry} = \frac{t_{stance}}{2}v + k(v - v ^{cmd})
$$

The centrifugal term is omitted. $p_k$ is body position at k-timestep. $l_i$ is the shoulder position for $i^{th}$ leg with respect to local frame. $R_z(\Psi_k)$ is the rotation matrix translating velocity to global frame. $t_{stance}$ is time cycle and $k=0.03$ is the feedback gain.

However, we choose the footholds solely based on quantitative score from various aspects (distance current pos, terrain variance/gradient, support area etc.), rather than solving a optimization problem. 



### DRL

We use the framework from isaac-gym, with PPO algorithm.  With the following feature added:

- Remove teacher-student framework

- Add GRU and CE-net as terrain encoder. Latent dimension was increased from 64 to 512. 

- TODO: symmetric data augmentation 

  

**To integrate the foothold into DRL**, the relative position to the optimized foothold was fed as observations for both actor and critic network. Moreover, a sparse **reward** term was also added, which will be triggered in the touch-down time. 



Estimated training time is 10 hours. 



### Set up


```shell
pip install -e rsl_rl

pip install -e .
```


### 
