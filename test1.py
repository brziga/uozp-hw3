import sys
import numpy as np
from vispy import app, scene

# Create a canvas with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()

# Define points and lines between them
points = np.array([
    [0, 0, 0],  # Origin
    [1, 2, 3],  # Point 1
    [2, 0, -1], # Point 2
    [-1, -1, 1] # Point 3
])

# Define line connections (start and end indices)
lines_index = np.array([
    [0, 1],
    [0, 2],
    [0, 3]
])

# Create the line visual
line_visual = scene.visuals.Line(pos=points, connect=lines_index, color='yellow', method='gl', parent=view.scene)

# Add labels at each line end point (excluding the origin)
labels = ['Point 1', 'Point 2', 'Point 3']
for i, label in enumerate(labels, start=1):
    label_visual = scene.visuals.Text(text=label, pos=points[i], color='white', font_size=14, parent=view.scene)

# Configure the camera to look at the 3D scatter plot
view.camera = 'turntable'  # 'arcball' for more interactive manipulation

# Add XYZ axes for better orientation
axis = scene.visuals.XYZAxis(parent=view.scene)

# Run the application
if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()
