# RaceUtils

<table>
  <tr>
    <td style="width:50%;"><img src="docs/cover/race_uzh_19g_3d.png" alt="race_uzh_19g_3d" style="width:100%;"/></td>
    <td style="width:50%;"><img src="docs/cover/figure8_3d.png" alt="figure8_3d" style="width:100%;"/></td>
  </tr>
</table>

**RaceUtils** is a Python package that provides tools for creating, manipulating, and visualizing drone race tracks. Whether you need predefined professional tracks, randomly generated courses, or custom trajectories, this library offers a comprehensive solution for drone racing simulation and training environments. If you want to learn more, refer to [Tools](#tools).

## Installations

For the latest development version or to contribute:

```bash
# Clone the repository
git clone https://github.com/KafuuChikai/RaceUtils.git
cd RaceUtils

# Install in development mode
pip install -e .
```

## Visualizations

### 1. Predefined trajectory

You can get a predefined trajectory below:

<p>
  <img src="docs/predefined_trajectory/plan_race_example.png" alt="plan_race_example" width="50%" />
</p>

### 2. Random trajectories

The random trajectory examples:

<table>
  <tr>
    <td style="width:50%;"><img src="docs/random_race_example/race_1.png" alt="race_1" style="width:100%;"/></td>
    <td style="width:50%;"><img src="docs/random_race_example/race_2.png" alt="race_2" style="width:100%;"/></td>
  </tr>
</table>

<table>
  <tr>
    <td style="width:50%;"><img src="docs/random_race_example/race_3.png" alt="race_3" style="width:100%;"/></td>
    <td style="width:50%;"><img src="docs/random_race_example/race_4.png" alt="race_4" style="width:100%;"/></td>
  </tr>
</table>

<table>
  <tr>
    <td style="width:50%;"><img src="docs/random_race_example/race_5.png" alt="race_5" style="width:100%;"/></td>
    <td style="width:50%;"><img src="docs/random_race_example/race_6.png" alt="race_6" style="width:100%;"/></td>
  </tr>
</table>

### 3. 3D trajectories with Tube

**Update**: Plot 3D trajectories and visulize with Tube.

<table>
  <tr>
    <td style="width:50%;"><img src="docs/3d_tube/random_example_2d.png" alt="random_example_2d" style="width:100%;"/></td>
    <td style="width:50%;"><img src="docs/3d_tube/random_example_3d.png" alt="random_example_3d" style="width:100%;"/></td>
  </tr>
</table>
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

## Tools

1. [RaceGenerator](docs/utils_manual.md#L3)
2. RaceVisualizer