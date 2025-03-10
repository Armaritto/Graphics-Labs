import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import random
# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW cannot be initialized!")

# Create the window
width, height = 800, 600
window = glfw.create_window(width, height, "Modern OpenGL Animation", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window cannot be created!")

# Make the context current
glfw.make_context_current(window)

# Define Vertex Shader
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 projection;
void main()
{
    gl_Position = projection * model * vec4(position, 1.0);
}
"""

# Define Fragment Shader
fragment_shader = """
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(0.0, 0.5, 1.0, 1.0); 
}
"""

# Compile Shaders
def create_shader_program():
    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    return shader
shader = create_shader_program()
# Define Circle Vertices
segments = 100
angle_step = 2 * np.pi / segments
radius = 0.1
vertices = [(0.0, 0.0, 0.0)]  # Center of the circle
for i in range(segments + 1):
    angle = i * angle_step
    x = np.cos(angle) * radius  # Ball radius
    y = np.sin(angle) * radius #* width / height
    vertices.append((x, y, 0.0))

vertices = np.array(vertices, dtype=np.float32)

# Upload data to GPU
vao = glGenVertexArrays(1)
vbo = glGenBuffers(1)

glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW) # GL_DYNAMIC_DRAW so it can be updated
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
glEnableVertexAttribArray(0)

# Projection Matrix
projection = pyrr.matrix44.create_orthogonal_projection(-1, 1, -1, 1, -1, 1)

# Animation Variables
ball_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
velocity = np.array([random.uniform(0, 1), random.uniform(0, 1), 0.0], dtype=np.float32)
velocity = velocity / np.linalg.norm(velocity)

scaling = np.array([1.0, 1.0])  # Initial scaling factors for x and y
scaling_speed = np.array([0.0, 0.0])  # Scaling speed for animation


# Key callback function
def key_callback(window, key, scancode, action, mods):
    global velocity, radius
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_RIGHT:
            velocity[0] *= 1.1
        elif key == glfw.KEY_LEFT:
            velocity[0] *= 0.9
        elif key == glfw.KEY_UP:
            velocity[1] *= 1.1
        elif key == glfw.KEY_DOWN:
            velocity[1] *= 0.9
        elif key == glfw.KEY_PAGE_UP:
            radius *= 1.05
        elif key == glfw.KEY_PAGE_DOWN:
            radius *= 0.95

glfw.set_key_callback(window, key_callback)

# Resize callback function
def framebuffer_size_callback(window, new_width, new_height):
    """ Adjusts the OpenGL viewport and updates the projection matrix. """
    global projection, width, height, ball_pos, vao, vbo
    width, height = new_width, new_height
    ball_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    vertices = [(0.0, 0.0, 0.0)]  # Center of the ball
    for i in range(segments + 1):
        angle = i * angle_step
        x = np.cos(angle) * radius  # Ball radius
        y = np.sin(angle) * radius * width / height
        vertices.append((x, y, 0.0))
    vertices = np.array(vertices, dtype=np.float32)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
    glEnableVertexAttribArray(0)
    glViewport(0, 0, width, height)

# Set the framebuffer resize callback
glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)



last_time = glfw.get_time()
# Rendering Loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    current_time = glfw.get_time()
    delta_time = current_time - last_time
    last_time = current_time
    # Update Ball Position
    ball_pos += velocity * delta_time

    # Collision detection
    if ball_pos[0] + radius > 1.0 or ball_pos[0] - radius < -1.0:
        velocity[0] = -velocity[0]
        scaling = np.array([0.8, 1.2])
        scaling_speed = np.array([0.1, -0.1])
    if ball_pos[1] + radius > 1.0 or ball_pos[1] - radius < -1.0:
        velocity[1] = -velocity[1]
        scaling = np.array([1.2, 0.8])
        scaling_speed = np.array([-0.1, 0.1])
    
    # Animate Scaling Back to Normal
    scaling += scaling_speed * delta_time * 10  
    scaling = np.clip(scaling, 0.8, 1.2)  # Limit the squeeze/stretch
    if np.allclose(scaling, [1.0, 1.0], atol=0.02):  
        scaling = np.array([1.0, 1.0])  # Reset to normal
        scaling_speed = np.array([0.0, 0.0])  # Stop animation
    
    # Model Transformation Matrix
    scale_matrix = pyrr.matrix44.create_from_scale([scaling[0], scaling[1], 1.0])
    translation_matrix = pyrr.matrix44.create_from_translation(ball_pos)
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
    glDrawArrays(GL_TRIANGLE_FAN, 0, len(vertices))

    glfw.swap_buffers(window)

# Cleanup
glfw.terminate()
