import numpy as np
import matplotlib.pyplot as plt

# Parameters
width = 960
height = 500
k = 30
radius = 50
cellSize = radius/sqrt(2)
gridWidth = np.ceil(width/cellSize)
gridHeight = np.ceil(height/cellSize)
grid = np.array(gridHeight * gridWidth)
queue = []
queueSize = 0

