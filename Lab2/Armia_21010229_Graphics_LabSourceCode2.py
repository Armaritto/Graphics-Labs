import glfw
from OpenGL.GL import *
import numpy as np
import pyrr
from OpenGL.GL.shaders import compileProgram, compileShader

def load_model(path):
    lines = []
    faces = []
    
    with open(path, 'r') as file:
        mode = ''
        n = 0
        i = 0
        current_line = []
        current_face = []
        
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue 
            
            if i == n: 
                if mode == 'l':
                    lines.append(current_line)
                elif mode == 'f':
                    faces.append(current_face)
                
                mode = parts[1]
                n = int(parts[0])
                i = 0
                current_line = []
                current_face = []
                
            else:
                x, y = float(parts[0]), float(parts[1])
                if mode == 'l':
                    current_line.append([x, y, 0.0])
                elif mode == 'f':
                    current_face.append([x, y, 0.0])
                i += 1

    line_vertices = []
    for line in lines:
        for vertex in line:
            line_vertices.extend(vertex)
            line_vertices.extend([1, 1, 1]) 

    face_vertices = []
    for face in faces:
        for vertex in face:
            face_vertices.extend(vertex)
            face_vertices.extend([1, 1, 1]) 

    return np.array(line_vertices, dtype=np.float32), np.array(face_vertices, dtype=np.float32)

# Vertex Shader Code
vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
out vec3 ourColor;
uniform mat4 model;
uniform mat4 projection;
void main()
{
    gl_Position = projection * model * vec4(aPos, 1.0);
    ourColor = aColor;
}
"""

# Fragment Shader Code
fragment_shader = """
#version 330 core
in vec3 ourColor;
out vec4 FragColor;
void main()
{
    FragColor = vec4(ourColor, 1.0);
}
"""

def create_shader_program():
    return compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )

def create_VAOs(vertices):
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return VAO, VBO, len(vertices) // 6

def main():
    if not glfw.init():
        return
    
    window = glfw.create_window(800, 600, "Snoopy OpenGL", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    
    line_vertices, face_vertices = load_model("snoopy3.txt")

    shader_program = create_shader_program()

    VAO_lines, VBO_lines, line_count = create_VAOs(line_vertices)
    VAO_faces, VBO_faces, face_count = create_VAOs(face_vertices)

    model_loc = glGetUniformLocation(shader_program, "model")
    projection_loc = glGetUniformLocation(shader_program, "projection")

    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)

        projection = pyrr.matrix44.create_orthogonal_projection_matrix(-50, 50, -50, 50, -10, 10)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

        model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

        glBindVertexArray(VAO_faces)
        glDrawArrays(GL_TRIANGLE_FAN, 0, face_count)
        glBindVertexArray(0)

        glBindVertexArray(VAO_lines)
        glDrawArrays(GL_LINE_STRIP, 0, line_count)
        glBindVertexArray(0)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
