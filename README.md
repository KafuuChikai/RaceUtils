# RaceUtils

|  |  |
| :---------- | :---------- |
| <video src="https://github.com/user-attachments/assets/4971501e-bc81-4c1a-aec7-e794125dc193" /> | <video src="https://github.com/user-attachments/assets/4211e1ae-d321-45bf-bf8a-52991d878eea" /> |

**RaceUtils** is a Python package that provides tools for creating, manipulating, and visualizing drone race tracks. Whether you need predefined professional tracks, randomly generated courses, or custom trajectories, this library offers a comprehensive solution for drone racing simulation and training environments. To learn more, refer to [Tools](#tools).

## Table of Contents

1. [Installations](#installations)
2. [Examples](#examples)
3. [Visualizations](#visualizations)
   - [3.1 Trajectories](#1-trajectories)
   - [3.2 Trajectories with tubes](#2-trajectories-with-tubes)
   - [3.3 Animations](#3-animations)
4. [Tools](#tools)

## Installations

For the latest development version or to contribute:

```bash
# Clone the repository
git clone https://github.com/KafuuChikai/RaceUtils.git
cd RaceUtils

# Install in development mode
pip install -e .
```

## Examples

Coming soon ...

## Visualizations

### 1. Trajectories

The random trajectory examples:

<table>
  <tr>
    <td style="width:50%;"><img src="docs/random_race_example/race_1.png" alt="race_1" style="width:100%;"/></td>
    <td style="width:50%;"><img src="docs/random_race_example/race_2.png" alt="race_2" style="width:100%;"/></td>
  </tr>
</table>

### 2. Trajectories with tubes

Plot 3D trajectories and visualize them with a tube.

<table>
  <tr>
    <td style="width:50%;"><img src="docs/3d_tube/figure8_2d.png" alt="figure8_2d" style="width:100%;"/></td>
    <td style="width:50%;"><img src="docs/3d_tube/figure8_3d.png" alt="figure8_3d" style="width:100%;"/></td>
  </tr>
</table>

<table>
  <tr>
    <td style="width:50%;"><img src="docs/3d_tube/race_uzh_19g_2d.png" alt="race_uzh_19g_2d" style="width:100%;"/></td>
    <td style="width:50%;"><img src="docs/3d_tube/race_uzh_19g_3d.png" alt="race_uzh_19g_3d" style="width:100%;"/></td>
  </tr>
</table>

### 3. Animations

Create animations for flight details:

|  |  |  |
| :---------- | :---------- | :---------- |
| <img src="docs/animations/Star_5_single_drone1_3d.png" /> | <img src="docs/animations/Star_5_single_drone1_3d.png" /> | <video src="https://github.com/user-attachments/assets/8e6eb8d0-df78-43c4-b1b2-a489ab879693" /> |
| <img src="docs/animations/UZH_single_drone1_3d.png" /> | <img src="docs/animations/UZH_single_drone1_3d.png" /> | <video src="https://github.com/user-attachments/assets/2e0c4502-95e2-4b02-94cf-f87b21c82db0" /> |

## Tools

1. [RaceGenerator](docs/utils_manual.md#L3)
2. RaceVisualizer
