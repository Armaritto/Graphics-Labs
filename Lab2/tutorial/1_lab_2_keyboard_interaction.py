import glfw
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
h, w = 1200, 1200
vertex_shader = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
out vec3 color;
void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    color = aColor;
}
"""

fragment_shader = """
#version 330 core
in vec3 color;
out vec4 FragColor;
void main()
{
    FragColor = vec4(color, 0.2);
}
"""

def create_shader_program():
    return compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )

def main():
    if not glfw.init():
        return

    window = glfw.create_window(w, h, "Modern OpenGL Keyboard Interaction", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    shader = create_shader_program()

    vertices = np.array([], dtype=np.float32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    color = (1.0, 1.0, 1.0)
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))
    glEnableVertexAttribArray(1)
    
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    def key_callback(window, key, scancode, action, mods):
        nonlocal VBO, vertices, color
        if action == glfw.PRESS:
            if key == glfw.KEY_R:
                print("R")
                color = (1.0, 0.0, 0.0)
            elif key == glfw.KEY_G:
                print("G")
                color = (0.0, 1.0, 0.0)
            elif key == glfw.KEY_B:
                print("B")
                color = (0.0, 0.0, 1.0)
            elif key == glfw.KEY_D:
                print("D")
                xpos, ypos = glfw.get_cursor_pos(window)
                print(xpos, ypos)
                xpos = (xpos / w) * 2 - 1
                ypos = ((h - ypos) / h) * 2 - 1
                r, g, b = color
                new_vertex = np.array([xpos, ypos, r, g, b], dtype=np.float32)
                vertices = np.concatenate((vertices, new_vertex))
            glBindBuffer(GL_ARRAY_BUFFER, VBO)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    glfw.set_key_callback(window, key_callback)
    glPointSize(50)
    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader)
        glBindVertexArray(VAO)
        glDrawArrays(GL_POINTS, 0, len(vertices) // 5)
        glBindVertexArray(0)
        glfw.swap_buffers(window)
    
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glfw.terminate()

if __name__ == "__main__":
    main()
