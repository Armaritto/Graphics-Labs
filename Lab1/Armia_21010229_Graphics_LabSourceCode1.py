import glfw
from OpenGL.GL import *
import numpy as np

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW could not be initialized!")

# Define window position variables
window_x = 0
window_y = 0

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(800, 600, "OpenGL Window", None, None)

if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window!")

# Set window position
glfw.set_window_pos(window, window_x, window_y)

# Make the window's context current
glfw.make_context_current(window)

# Define vertex data for a triangle
vertex1 = [0.6, -0.5, 0.0]
vertex2 = [-0.1, 0.9, 0.0]
vertex3 = [0.1, -0.9, 0.0]
vertices = np.array(vertex1 + vertex2 + vertex3, dtype=np.float32)

# Generate and bind a Vertex Buffer Object (VBO)
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Enable and define the vertex attribute pointer
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

# Unbind the VBO
glBindBuffer(GL_ARRAY_BUFFER, 0)

# Main render loop
while not glfw.window_should_close(window):
    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT)

    # Bind the VBO and draw the triangle
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glDisableVertexAttribArray(0)
    
    # Swap front and back buffers
    glfw.swap_buffers(window)

    # Poll for and process events
    glfw.poll_events()

# Cleanup and exit
glDeleteBuffers(1, [vbo])
glfw.terminate()