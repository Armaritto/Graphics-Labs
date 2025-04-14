import glfw
from OpenGL.GL import *
import numpy as np
from PIL import Image
import ctypes
import pyrr

# Vertex Shader
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoords;

out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    TexCoords = texCoords;
}
"""

# Fragment Shader
FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D texture1;

void main() {
    FragColor = texture(texture1, TexCoords);
}
"""

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW initialization failed!")

# Create a window
window = glfw.create_window(800, 600, "Modern OpenGL Texturing", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed!")

glfw.make_context_current(window)

# Enable depth testing for proper 3D rendering
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Compile shaders
def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_shader_program():
    program = glCreateProgram()
    vertex_shader = compile_shader(GL_VERTEX_SHADER, VERTEX_SHADER)
    fragment_shader = compile_shader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
    
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    
    return program

shader_program = create_shader_program()

# Define Cube Vertices with Proper Texture Coordinates
cube_vertices = np.array([
    # 3D Positions       # Texture Coords (U, V)

    # Front face
    -0.5, -0.5,  0.5,   0.0, 0.0,
     0.5, -0.5,  0.5,   1.0, 0.0,
     0.5,  0.5,  0.5,   1.0, 1.0,
    -0.5,  0.5,  0.5,   0.0, 1.0,

    # Back face
    -0.5, -0.5, -0.5,   1.0, 0.0,
    -0.5,  0.5, -0.5,   1.0, 1.0,
     0.5,  0.5, -0.5,   0.0, 1.0,
     0.5, -0.5, -0.5,   0.0, 0.0,

    # Left face
    -0.5, -0.5, -0.5,   0.0, 0.0,
    -0.5, -0.5,  0.5,   1.0, 0.0,
    -0.5,  0.5,  0.5,   1.0, 1.0,
    -0.5,  0.5, -0.5,   0.0, 1.0,

    # Right face
     0.5, -0.5, -0.5,   1.0, 0.0,
     0.5,  0.5, -0.5,   1.0, 1.0,
     0.5,  0.5,  0.5,   0.0, 1.0,
     0.5, -0.5,  0.5,   0.0, 0.0,

    # Top face
    -0.5,  0.5, -0.5,   0.0, 0.0,
    -0.5,  0.5,  0.5,   0.0, 1.0,
     0.5,  0.5,  0.5,   1.0, 1.0,
     0.5,  0.5, -0.5,   1.0, 0.0,

    # Bottom face
    -0.5, -0.5, -0.5,   0.0, 1.0,
     0.5, -0.5, -0.5,   1.0, 1.0,
     0.5, -0.5,  0.5,   1.0, 0.0,
    -0.5, -0.5,  0.5,   0.0, 0.0,
], dtype=np.float32)

# Define Proper Indices
indices = np.array([
    0, 1, 2, 2, 3, 0,  # Front
    4, 5, 6, 6, 7, 4,  # Back
    8, 9, 10, 10, 11, 8,  # Left
    12, 13, 14, 14, 15, 12,  # Right
    16, 17, 18, 18, 19, 16,  # Top
    20, 21, 22, 22, 23, 20   # Bottom
], dtype=np.uint32)


# Load and Bind Texture
def load_texture(path):
    image = Image.open(path)
    # image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(image.convert("RGB"), dtype=np.uint8)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    return texture

texture = load_texture("assets/textures/sample1.jpg")

# Set up VAO, VBO, and EBO
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)
EBO = glGenBuffers(1)

glBindVertexArray(VAO)

glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)

glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

# Position attribute
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * cube_vertices.itemsize, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# Texture coordinate attribute
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * cube_vertices.itemsize, ctypes.c_void_p(3 * cube_vertices.itemsize))
glEnableVertexAttribArray(1)

glBindVertexArray(0)

# Create transformation matrices
projection = pyrr.matrix44.create_perspective_projection(45, 800/600, 0.1, 100, dtype=np.float32)
view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, -3]), dtype=np.float32)

# Main render loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(shader_program)

    model = pyrr.matrix44.create_from_eulers(
        [glfw.get_time(), glfw.get_time(), 0], dtype=np.float32
    )

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    proj_loc = glGetUniformLocation(shader_program, "projection")

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    glBindTexture(GL_TEXTURE_2D, texture)
    glBindVertexArray(VAO)
    
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    
    glfw.swap_buffers(window)

glfw.terminate()
