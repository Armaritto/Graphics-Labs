import glfw
from OpenGL.GL import *
import numpy as np
import pyrr
import math

# Circle vertex generator
def create_circle_vertices(radius=1.0, segments=64):
    verts = [(0.0, 0.0, 0.0)]  # center
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        verts.append((x, y, 0.0))
    return np.array(verts, dtype=np.float32)

# Shader source
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

# Setup GLFW
if not glfw.init():
    raise Exception("GLFW init failed")

window = glfw.create_window(800, 600, "2D Solar System with Circles", None, None)
glfw.make_context_current(window)

# Compile shader
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

# Generate circle data
circle_vertices = create_circle_vertices()
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)
glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, circle_vertices.nbytes, circle_vertices, GL_STATIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# Uniform locations
transform_loc = glGetUniformLocation(shader, "transform")
color_loc = glGetUniformLocation(shader, "color")

# Define planets [radius, speed, scale, color]
planets = [
    {"radius": 0.2, "speed": 1.0, "scale": 0.03, "color": (0.5, 0.5, 1.0)},   # Mercury
    {"radius": 0.4, "speed": 0.6, "scale": 0.05, "color": (1.0, 0.5, 0.2)},   # Venus
    {"radius": 0.6, "speed": 0.4, "scale": 0.07, "color": (0.2, 0.8, 0.2)},   # Earth
    {"radius": 0.8, "speed": 0.2, "scale": 0.09, "color": (0.8, 0.3, 0.6)}    # Mars
]

# Main render loop
glClearColor(0, 0, 0, 1)
while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT)

    time = glfw.get_time()

    glBindVertexArray(VAO)

    # Draw sun
    glUniform3f(color_loc, 1.0, 1.0, 0.0)  # Yellow
    sun_transform = pyrr.matrix44.create_from_scale([0.12]*3, dtype=np.float32)
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, sun_transform)
    glDrawArrays(GL_TRIANGLE_FAN, 0, len(circle_vertices))

    # Draw planets
    for planet in planets:
        angle = time * planet["speed"]
        x = planet["radius"] * math.cos(angle)
        y = planet["radius"] * math.sin(angle)

        translation = pyrr.matrix44.create_from_translation([x, y, 0.0])
        scale = pyrr.matrix44.create_from_scale([planet["scale"]]*3)
        transform = pyrr.matrix44.multiply(scale, translation)

        glUniform3f(color_loc, *planet["color"])
        glUniformMatrix4fv(transform_loc, 1, GL_FALSE, transform)
        glDrawArrays(GL_TRIANGLE_FAN, 0, len(circle_vertices))

    glfw.swap_buffers(window)

glfw.terminate()
