import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from pyrr import Matrix44, Vector3, matrix44

VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(0.0, 0.0, 0.0, 1.0);  // black
}
"""

cube_vertices = np.array([
    # positions only
    -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5,  0.5, -0.5,
    -0.5,  0.5, -0.5,
    -0.5, -0.5,  0.5,
     0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
], dtype=np.float32)

cube_indices = np.array([
    # lines to form wireframe cube
    0, 1, 1, 2, 2, 3, 3, 0,
    4, 5, 5, 6, 6, 7, 7, 4,
    0, 4, 1, 5, 2, 6, 3, 7
], dtype=np.uint32)

def window_setup():
    if not glfw.init():
        raise Exception("GLFW init failed")
    window = glfw.create_window(800, 600, "OpenGL Cube", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed")
    glfw.make_context_current(window)
    return window

def create_shader_program():
    return compileProgram(
        compileShader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)
    )

def create_cube_vao():
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.nbytes, cube_indices, GL_STATIC_DRAW)

    # position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)
    return vao

def main():
    window = window_setup()
    shader = create_shader_program()
    cube_vao = create_cube_vao()

    glEnable(GL_DEPTH_TEST)
    t = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        model = matrix44.multiply(matrix44.multiply(Matrix44.from_scale([1.0, 1.0, 5.0]), Matrix44.from_z_rotation(np.radians(t))), Matrix44.from_translation([1.0, 1.0, 1.0]))

        view = Matrix44.look_at(
            eye=Vector3([3.0, 6.0, 5.0]),
            target=Vector3([1.0, 0.0, 0.0]),
            up=Vector3([0.0, 0.0, 1.0])
        )

        projection = Matrix44.perspective_projection(45.0, 800/600, 0.1, 100.0)

        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model.astype(np.float32))
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view.astype(np.float32))
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection.astype(np.float32))

        glBindVertexArray(cube_vao)
        glDrawElements(GL_LINES, len(cube_indices), GL_UNSIGNED_INT, None)
        t += 30 * glfw.get_time()
        glfw.set_time(0)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
