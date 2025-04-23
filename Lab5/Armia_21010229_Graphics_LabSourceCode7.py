import glfw
from OpenGL.GL import *
import numpy as np
import pyrr
import math

vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
uniform mat4 transform;
void main()
{
    gl_Position = transform * vec4(a_position, 1.0);
}
"""

fragment_src = """
# version 330
uniform vec3 color;
out vec4 out_color;
void main()
{
    out_color = vec4(color, 1.0);
}
"""

if not glfw.init():
    raise Exception("GLFW init failed")

window = glfw.create_window(800, 600, "2D Robotic Arm", None, None)
glfw.make_context_current(window)

def compile_shader(src, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

shader = glCreateProgram()
glAttachShader(shader, compile_shader(vertex_src, GL_VERTEX_SHADER))
glAttachShader(shader, compile_shader(fragment_src, GL_FRAGMENT_SHADER))
glLinkProgram(shader)
glUseProgram(shader)

vertices = np.array([
    [0.0, 0.0, 0.0], 
    [0.4, 0.0, 0.0],  
    [0.4, 0.4, 0.0],  
    [0.4, 0.5, 0.0]  
], dtype=np.float32)

vertices = vertices.flatten()
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)
glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

transform_loc = glGetUniformLocation(shader, "transform")
color_loc = glGetUniformLocation(shader, "color")

shoulder_angle = math.radians(90)
elbow_angle = 0
wrist_angle = 0

def create_rotation_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

SHOULDER_MIN = math.radians(90)
SHOULDER_MAX = math.radians(270)

ELBOW_MIN = math.radians(0)
ELBOW_MAX = math.radians(145)

WRIST_MIN = math.radians(-90)
WRIST_MAX = math.radians(90)

def key_callback(window, key, scancode, action, mods):
    global shoulder_angle, elbow_angle, wrist_angle
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_Q: 
            shoulder_angle += 0.1
        elif key == glfw.KEY_W: 
            shoulder_angle -= 0.1
        elif key == glfw.KEY_A: 
            elbow_angle += 0.1
        elif key == glfw.KEY_S: 
            elbow_angle -= 0.1
        elif key == glfw.KEY_Z:  
            wrist_angle += 0.1
        elif key == glfw.KEY_X: 
            wrist_angle -= 0.1

    shoulder_angle = max(min(shoulder_angle, SHOULDER_MAX), SHOULDER_MIN)
    elbow_angle = max(min(elbow_angle, ELBOW_MAX), ELBOW_MIN)
    wrist_angle = max(min(wrist_angle, WRIST_MAX), WRIST_MIN)

glfw.set_key_callback(window, key_callback)

glClearColor(0, 0, 0, 1)

stack = []

def push_matrix(matrix):
    stack.append(np.copy(matrix))

def pop_matrix():
    return stack.pop()

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT)

    base_transform = np.eye(4, dtype=np.float32)  

    shoulder_transform = pyrr.matrix44.multiply(
        create_rotation_matrix(shoulder_angle),
        base_transform
    )
    push_matrix(shoulder_transform)
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, shoulder_transform)
    glUniform3f(color_loc, 1.0, 0.0, 0.0) 
    glDrawArrays(GL_LINES, 0, 2)

    elbow_transform = pyrr.matrix44.multiply(
        create_rotation_matrix(elbow_angle),
        pyrr.matrix44.create_from_translation([0.4, 0.0, 0.0])
    )
    elbow_transform = pyrr.matrix44.multiply(elbow_transform, stack[-1])
    push_matrix(elbow_transform)
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, elbow_transform)
    glUniform3f(color_loc, 0.0, 1.0, 0.0) 
    glDrawArrays(GL_LINES, 0, 2)

    wrist_transform = pyrr.matrix44.multiply(
        create_rotation_matrix(wrist_angle),
        pyrr.matrix44.create_from_translation([0.4, 0.0, 0.0])
    )
    wrist_transform = pyrr.matrix44.multiply(wrist_transform, stack[-1])
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, wrist_transform)
    glUniform3f(color_loc, 0.0, 0.0, 1.0)  
    glDrawArrays(GL_LINES, 0, 2)

    pop_matrix()  
    pop_matrix()

    glfw.swap_buffers(window)

glfw.terminate()