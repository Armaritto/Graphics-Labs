import glfw
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader

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
    FragColor = vec4(color, 1.0);
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

    window = glfw.create_window(1000, 1000, "Modern OpenGL Mouse Interaction", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    shader = create_shader_program()

    vertices = np.array([], dtype=np.float32)
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))
    glEnableVertexAttribArray(1)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)  # Set white background
    
    def mouse_callback(window, button, action, mods):
        nonlocal vertices, VBO
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)
            print(f"Left Mouse Click: {xpos}, {ypos}")
            
            xpos = (xpos / 1000) * 2 - 1
            ypos = ((1000 - ypos) / 1000) * 2 - 1
            new_vertex = np.array([xpos, ypos, 0.0, 0.0, 0.0], dtype=np.float32)  # Black color
            vertices = np.concatenate((vertices, new_vertex))
            glBindBuffer(GL_ARRAY_BUFFER, VBO)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    glfw.set_mouse_button_callback(window, mouse_callback)
    glPointSize(15)
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
