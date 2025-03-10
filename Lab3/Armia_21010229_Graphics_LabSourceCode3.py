import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import random

if not glfw.init():
    raise Exception("GLFW cannot be initialized!")

width, height = 800, 600
window = glfw.create_window(width, height, "Bounce Ball", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window cannot be created!")

glfw.make_context_current(window)

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

fragment_shader = """
#version 330 core
out vec4 FragColor;
uniform vec4 color;
void main()
{
    FragColor = color;
}
"""

def create_shader_program():
    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    return shader
shader = create_shader_program()

segments = 100
angle_step = 2 * np.pi / segments
radius = 0.03

vertices = [(0.0, 0.0, 0.0)]
for i in range(segments + 1):
    angle = i * angle_step
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius * width / height
    vertices.append((x, y, 0.0))

vertices = np.array(vertices, dtype=np.float32)

vao = glGenVertexArrays(1)
vbo = glGenBuffers(1)

glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
glEnableVertexAttribArray(0)

width2 = 0.2
height2 = 0.05
vertices2 = [
    (-width2 / 2, -height2 / 2, 0.0),
    (width2 / 2, -height2 / 2, 0.0),
    (width2 / 2, height2 / 2, 0.0),
    (-width2 / 2, height2 / 2, 0.0)
]
vertices2 = np.array(vertices2, dtype=np.float32)

vao2 = glGenVertexArrays(1)
vbo2 = glGenBuffers(1)

glBindVertexArray(vao2)
glBindBuffer(GL_ARRAY_BUFFER, vbo2)
glBufferData(GL_ARRAY_BUFFER, vertices2.nbytes, vertices2, GL_DYNAMIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
glEnableVertexAttribArray(0)

projection = pyrr.matrix44.create_orthogonal_projection(-1, 1, -1, 1, -1, 1)

ball_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
bar_pos = np.array([0.0, -0.8, 0.0], dtype=np.float32)
velocity = np.array([0, 0, 0.0], dtype=np.float32)
gravity = np.array([0.0, -0.98, 0.0], dtype=np.float32)

ball_scaling = np.array([1.0, 1.0])
ball_scaling_speed = np.array([0.0, 0.0]) 

ball_color = np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float32)
bar_color = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32) 

def cursor_position_callback(window, xpos, ypos):
    xpos = (xpos / width) * 2 - 1

    global bar_pos
    bar_pos = np.array([xpos, -0.8, 0.0], dtype=np.float32)
    
glfw.set_cursor_pos_callback(window, cursor_position_callback)

def key_callback(window, key, scancode, action, mods):
    global bar_pos
    if key == glfw.KEY_LEFT and (action == glfw.PRESS or action == glfw.REPEAT):
        bar_pos[0] -= 0.05
    elif key == glfw.KEY_RIGHT and (action == glfw.PRESS or action == glfw.REPEAT):
        bar_pos[0] += 0.05
glfw.set_key_callback(window, key_callback)

def framebuffer_size_callback(window, new_width, new_height):
    global projection, width, height, ball_pos, vao, vbo
    width, height = new_width, new_height
    ball_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    vertices = [(0.0, 0.0, 0.0)] 
    for i in range(segments + 1):
        angle = i * angle_step
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius * width / height
        vertices.append((x, y, 0.0))
    vertices = np.array(vertices, dtype=np.float32)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)
    glEnableVertexAttribArray(0)
    glViewport(0, 0, width, height)

glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

last_time = glfw.get_time()

while not glfw.window_should_close(window):
    glfw.poll_events()
    current_time = glfw.get_time()
    delta_time = current_time - last_time
    last_time = current_time
    
    velocity += gravity * delta_time

    ball_pos += velocity * delta_time

    if ball_pos[0] + radius > 1.0 or ball_pos[0] - radius < -1.0:
        velocity[0] = -velocity[0]
        ball_scaling = np.array([0.8, 1.2])
        ball_scaling_speed = np.array([0.1, -0.1])
        
    if ball_pos[1] - radius < -0.8 and ball_pos[0] > bar_pos[0] - width2 / 2 and ball_pos[0] < bar_pos[0] + width2 / 2:
        velocity[1] = -velocity[1]
        velocity[0] = random.uniform(-0.5, 0.5)
        ball_scaling = np.array([1.2, 0.8])
        ball_scaling_speed = np.array([-0.1, 0.1])

    elif ball_pos[1] - radius < -1.0:
        ball_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        velocity = np.array([0, 0, 0.0], dtype=np.float32)

    ball_scaling += ball_scaling_speed * delta_time * 10  
    ball_scaling = np.clip(ball_scaling, 0.8, 1.2)  
    if np.allclose(ball_scaling, [1.0, 1.0], atol=0.02):  
        ball_scaling = np.array([1.0, 1.0])  
        ball_scaling_speed = np.array([0.0, 0.0])  
    
    ball_scale_matrix = pyrr.matrix44.create_from_scale([ball_scaling[0], ball_scaling[1], 1.0])
    ball_translation_matrix = pyrr.matrix44.create_from_translation(ball_pos)
    ball_model = pyrr.matrix44.multiply(ball_scale_matrix, ball_translation_matrix)

    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shader)

    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    color_loc = glGetUniformLocation(shader, "color")

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, ball_model)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniform4fv(color_loc, 1, ball_color)

    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_FAN, 0, len(vertices))

    bar_translation_matrix = pyrr.matrix44.create_from_translation(bar_pos)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, bar_translation_matrix)
    glUniform4fv(color_loc, 1, bar_color)

    glBindVertexArray(vao2)
    glDrawArrays(GL_QUADS, 0, len(vertices2))

    glfw.swap_buffers(window)

glfw.terminate()