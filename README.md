# medscope
Medical Intraoperative Real-time Visualization System.

## Demo

![](/img/demo.gif)

## Installation

> [!IMPORTANT]
> You should use `python >=3.10, <3.13`

```bash
pip install medscope
```

## Usage

> [!WARNING]
> Do not use multithreading, use `MedScopeWindow.add_timer`

```python
from medscope import MedScopeWindow, MedScopeSystem
import numpy as np
import sys

# Generate a random 3x3 rotation matrix uniformly distributed over SO(3).
def random_rotation_matrix() -> np.ndarray:
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)
    psi = np.random.uniform(0, 2 * np.pi)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    R = np.array([
        [cos_phi*cos_theta*cos_psi - sin_phi*sin_psi, 
         -cos_phi*cos_theta*sin_psi - sin_phi*cos_psi, 
         cos_phi*sin_theta],
        [sin_phi*cos_theta*cos_psi + cos_phi*sin_psi, 
         -sin_phi*cos_theta*sin_psi + cos_phi*cos_psi, 
         sin_phi*sin_theta],
        [-sin_theta*cos_psi, 
         sin_theta*sin_psi, 
         cos_theta]
    ])
    
    return R

# Initialize app and window
app = MedScopeSystem(sys.argv)
window = MedScopeWindow()

# Add a 3D model
window.add_model_from_file(
    "bone_model",
    "BONE-1.new.stl",
    (1.0, 1.0, 1.0))  # white, random if not given

# Set the pose of camera
#   you can change camera pos in callback function with add_timer
window.set_camera_pose(
    (0, 0, -500),
    (0, 0, 0),
    (0, 1, 0)
)

# Volume data should be given in 3D np.ndarray 
#   and dtype should be np.uint8
#   RGB channel image 3 * N * M * L
window.set_volume(np.random.randint(0, 255, (3, 256, 256, 256)).astype(np.uint8))

# use N * M * L to achieve grey image
# window.set_volume(np.random.randint(0, 255, (256, 256, 256)).astype(np.uint8))

# Create a callback function to move your model
def move_model():
    import random
    x = random.randint(0, 256)
    y = random.randint(0, 256)
    z = random.randint(0, 256)
    window.set_slice_positions(x, y, z)
    window.set_model_pose(
        "bone_model",
        (x - 128, y - 128, z - 128),
        random_rotation_matrix()
    )

# Call move_model every 1ms (as quickly as the processor can)
window.add_timer("move_model", 1, move_model)
sys.exit(app.exec_())

```
