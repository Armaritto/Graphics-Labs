import glfw
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
import random
# Vertex Shader
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

# Fragment Shader
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

    window = glfw.create_window(800, 600, "Modern OpenGL Mouse Pen", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    shader = create_shader_program()
    vertices_list = []
    vao_list = []
    vbo_list = []
    
    
    glClearColor(1.0, 1.0, 1.0, 1.0)  # White background
    mouse_held = False

     # Mouse button callback function
    def mouse_button_callback(window, button, action, mods):
        nonlocal mouse_held, vertices_list, vao_list, vbo_list
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                if not mouse_held:
                    vertices_list.append(np.array([], dtype=np.float32))
                    vao_list.append(glGenVertexArrays(1))
                    vbo_list.append(glGenBuffers(1))

                    glBindVertexArray(vao_list[-1])
                    glBindBuffer(GL_ARRAY_BUFFER, vbo_list[-1])
                    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
                    glEnableVertexAttribArray(0)
                    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))
                    glEnableVertexAttribArray(1)
                mouse_held = True
            elif action == glfw.RELEASE:
                mouse_held = False
    
    # Mouse movement callback function
    def mouse_move_callback(window, xpos, ypos):
        nonlocal vertices_list, vbo_list, mouse_held
        if mouse_held:
            xpos = (xpos / 800) * 2 - 1
            ypos = ((600 - ypos) / 600) * 2 - 1
            new_vertex = np.array([xpos, ypos, random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], dtype=np.float32)  # Black color
            vertices_list[-1] = np.concatenate((vertices_list[-1], new_vertex))
            glBindBuffer(GL_ARRAY_BUFFER, vbo_list[-1])
            glBufferData(GL_ARRAY_BUFFER, vertices_list[-1].nbytes, vertices_list[-1], GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, mouse_move_callback)
    glLineWidth(5)
    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader)
        for vao, vertices in zip(vao_list, vertices_list):
            glBindVertexArray(vao)
            glDrawArrays(GL_LINE_STRIP, 0, len(vertices) // 5)
            glBindVertexArray(0)
        glfw.swap_buffers(window)
    for vao, vbo in zip(vao_list, vbo_list):
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
    glfw.terminate()

if __name__ == "__main__":
    main()
