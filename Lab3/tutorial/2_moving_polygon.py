import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader  # Import shader compilation utilities

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW cannot be initialized!")

# Window size
width, height = 500, 500

# Create the window
window = glfw.create_window(width, height, "Moving Polygon", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window cannot be created!")

# Make the context current
glfw.make_context_current(window)

# Define Vertex Shader
vertex_shader = """
#version 330 core
layout(location = 0) in vec2 position;
uniform mat4 model;
uniform mat4 projection;
void main()
{
    gl_Position = projection * model * vec4(position, 0.0, 1.0);
}
"""

# Define Fragment Shader
fragment_shader = """
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);  // Polygon Color
}
"""

# Shader Compilation Function
def create_shader_program():
    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    return shader

# Compile shaders using the function
shader = create_shader_program()

# Define Polygon Vertices (Square)
size = 100
vertices = np.array([
    [0, 0],
    [size, 0],
    [size, -size],
    [0, -size]
], dtype=np.float32).flatten()

# Upload Data to GPU
vao = glGenVertexArrays(1)
vbo = glGenBuffers(1)

glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
glEnableVertexAttribArray(0)

# Initialize Projection Matrix
global projection
projection = pyrr.matrix44.create_orthogonal_projection(0, width, 0, height, -1, 1)

def update_projection_matrix(win, w, h):
    global projection, width, height
    width, height = w, h
    # aspect_ratio = w / h if h > 0 else 1
    projection = pyrr.matrix44.create_orthogonal_projection(0, w, 0, h, -1, 1)
    glViewport(0, 0, w, h)

glfw.set_framebuffer_size_callback(window, update_projection_matrix)

# Movement Variables
x_pos, y_pos = width / 2, height / 2
velocity = np.array([200, 190], dtype=np.float32)
scaling = np.array([1.0, 1.0])  # Initial scaling factors for x and y
scaling_speed = np.array([0.0, 0.0])  # Scaling speed for animation

# Frame Timing
last_time = glfw.get_time()

# Rendering Loop
while not glfw.window_should_close(window):
    glfw.poll_events()

    # Compute delta time
    current_time = glfw.get_time()
    delta_time = current_time - last_time
    last_time = current_time

    # Update Position (time-based movement)
    x_pos += velocity[0] * delta_time
    y_pos += velocity[1] * delta_time

    # Collision detection (handle resizing dynamically)
    if x_pos <= 0 or x_pos + size >= width:
        velocity[0] = -velocity[0]
        scaling = np.array([.8, 1.2])
        scaling_speed = np.array([0.1, -0.1])  # Animation speed for recovery
    
    if y_pos - size <= 0 or y_pos >= height:
        velocity[1] = -velocity[1]
        scaling = np.array([1.2, 0.8])
        scaling_speed = np.array([-0.1, 0.1])  # Animation speed for recovery

    # Animate Scaling Back to Normal
    scaling += scaling_speed * delta_time * 10  
    scaling = np.clip(scaling, 0.8, 1.2)  # Limit the squeeze/stretch
    if np.allclose(scaling, [1.0, 1.0], atol=0.02):  
        scaling = np.array([1.0, 1.0])  # Reset to normal
        scaling_speed = np.array([0.0, 0.0])  # Stop animation

    # Model Transformation Matrix (Translation + Scaling)
    scale_matrix = pyrr.matrix44.create_from_scale([scaling[0], scaling[1], 1.0])
    translation_matrix = pyrr.matrix44.create_from_translation([x_pos, y_pos, 0])
    model = pyrr.matrix44.multiply(scale_matrix, translation_matrix)

    # Clear screen
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shader)

    # Pass matrices to shader
    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    # Render
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

    glfw.swap_buffers(window)

# Cleanup
glfw.terminate()
