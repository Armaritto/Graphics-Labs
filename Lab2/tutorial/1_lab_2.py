import glfw
from OpenGL.GL import *
import numpy as np
import pyrr
from OpenGL.GL.shaders import compileProgram, compileShader
def load_model(path):
    vertices = []
    normals = []
    faces = []
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(v) for v in parts[1:4]])
            elif parts[0] == 'vn':
                normals.append([float(v) for v in parts[1:4]])
            elif parts[0] == 'f':
                face = []
                for part in parts[1:]:
                    indices = part.split('/')
                    vertex_index = int(indices[0]) - 1
                    normal_index = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else 0
                    face.append((vertex_index, normal_index))
                faces.append(face)

    # Expand faces into flat vertex data
    model_data = []
    for face in faces:
        for vertex_index, normal_index in face:
            model_data.extend(vertices[vertex_index])
            model_data.extend(normals[normal_index])

    return np.array(model_data, dtype=np.float32)

# Vertex Shader Code
vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 projection;
void main()
{
    gl_Position = projection * model * vec4(aPos, 1.0);
}
"""

# Fragment Shader Code
fragment_shader = """
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(0.5, 0.8, 0.2, 1.0); // Green color
}
"""

def create_shader_program():
    shader = compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )
    return shader

def create_model(path):
    model_vertices = load_model(path)
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, model_vertices.nbytes, model_vertices, GL_STATIC_DRAW)
        
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return VAO, len(model_vertices) // 5

def create_cube():
    vertices = np.array([
        -0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,
         0.5,  0.5, -0.5,
        -0.5,  0.5, -0.5,
        -0.5, -0.5,  0.5,
         0.5, -0.5,  0.5,
         0.5,  0.5,  0.5,
        -0.5,  0.5,  0.5
    ], dtype=np.float32)
    
    indices = np.array([
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 1, 5, 5, 4, 0,
        2, 3, 7, 7, 6, 2,
        0, 3, 7, 7, 4, 0,
        1, 2, 6, 6, 5, 1
    ], dtype=np.uint32)
    
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)
    
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    return VAO, EBO, len(indices)

def key_callback(window, key, scancode, action, mods):
    global projection_mode
    if action == glfw.PRESS:
        if key == glfw.KEY_P:
            projection_mode = "perspective"
        elif key == glfw.KEY_O:
            projection_mode = "orthographic"

def main():
    global projection_mode
    projection_mode = "perspective"
    
    if not glfw.init():
        return
    
    window = glfw.create_window(1600, 1200, "Lab 2 - OpenGL Cube", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    
    shader_program = create_shader_program()
    VAO, EBO, index_count = create_cube()
    VAO2, model_count = create_model("assests\\model_objects\\rifle.obj")
    glUseProgram(shader_program)
    model_loc = glGetUniformLocation(shader_program, "model")
    projection_loc = glGetUniformLocation(shader_program, "projection")
    
    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        
        if projection_mode == "perspective":
            projection = pyrr.Matrix44.perspective_projection(45.0, 800 / 600, 0.1, 100.0)
        else:
            projection = pyrr.Matrix44.orthogonal_projection(-2, 2, -2, 2, 0.1, 100.0)
        
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
        
        model_t = pyrr.Matrix44.from_y_rotation(glfw.get_time()) @ pyrr.Matrix44.from_translation([0, 0, -3])
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_t)
        
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        model_t = pyrr.Matrix44.identity() @ pyrr.Matrix44.from_translation([0, 0, -5])
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_t)
        glBindVertexArray(VAO2)
        glDrawArrays(GL_TRIANGLES, 0, model_count)
        glBindVertexArray(0)
        
        glfw.swap_buffers(window)
    
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [EBO])
    glfw.terminate()
    
if __name__ == "__main__":
    main()
