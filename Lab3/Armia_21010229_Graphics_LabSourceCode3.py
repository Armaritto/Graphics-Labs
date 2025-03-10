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