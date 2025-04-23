import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from pyrr import Matrix44, Vector3, matrix44
import ctypes
import time

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
    0, 1, 2, 3, 0,
    4, 5, 6, 7, 4,
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

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)
    return vao

def handle_input(window, camera_pos, camera_target, camera_up, camera_rot, delta_time):
    move_speed = 5.0 * delta_time
    rot_speed = 60.0 * delta_time

    forward = (camera_target - camera_pos).normalized
    right = np.cross(forward, camera_up).astype(np.float32)

    # Translation
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera_pos += forward * move_speed
        camera_target += forward * move_speed
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera_pos -= forward * move_speed
        camera_target -= forward * move_speed
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera_pos -= right * move_speed
        camera_target -= right * move_speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera_pos += right * move_speed
        camera_target += right * move_speed
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        camera_pos -= camera_up * move_speed
        camera_target -= camera_up * move_speed
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        camera_pos += camera_up * move_speed
        camera_target += camera_up * move_speed

    # Rotation
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera_rot[0] += rot_speed
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera_rot[0] -= rot_speed
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        camera_rot[1] += rot_speed
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        camera_rot[1] -= rot_speed
    if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS:
        camera_rot[2] += rot_speed
    if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS:
        camera_rot[2] -= rot_speed

    return camera_pos, camera_target, camera_rot

def main():
    window = window_setup()
    shader = create_shader_program()
    cube_vao = create_cube_vao()

    glEnable(GL_DEPTH_TEST)

    camera_pos = Vector3([3.0, 6.0, 5.0])
    camera_target = Vector3([1.0, 0.0, 0.0])
    camera_up = Vector3([0.0, 0.0, 1.0])
    camera_rot = Vector3([0.0, 0.0, 0.0])
    polygon_mode = GL_LINE
    glPolygonMode(GL_FRONT_AND_BACK, polygon_mode)

    glfw.set_input_mode(window, glfw.STICKY_KEYS, GL_TRUE)

    last_time = time.time()
    t = 0
    while not glfw.window_should_close(window):
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        glfw.poll_events()
        camera_pos, camera_target, camera_rot = handle_input(
            window, camera_pos, camera_target, camera_up, camera_rot, delta_time
        )

        if glfw.get_key(window, glfw.KEY_M) == glfw.PRESS:
            print(polygon_mode)
            polygon_mode = GL_FILL if polygon_mode == GL_LINE else GL_LINE
            glPolygonMode(GL_FRONT_AND_BACK, polygon_mode)

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        model = matrix44.multiply(matrix44.multiply(Matrix44.from_z_rotation(np.radians(t)), Matrix44.from_scale([1.0, 1.0, 4.0])), Matrix44.from_translation([1.0, 1.0, 1.0]))

        view = Matrix44.look_at(
            eye=camera_pos,
            target=camera_target,
            up=camera_up
        )

        rotation_matrix = matrix44.multiply(matrix44.multiply(Matrix44.from_x_rotation(np.radians(camera_rot.x)),
                          Matrix44.from_y_rotation(np.radians(camera_rot.y))),
                          Matrix44.from_z_rotation(np.radians(camera_rot.z)))
        view = matrix44.multiply(rotation_matrix, view)

        projection = Matrix44.perspective_projection(45.0, 800 / 600, 0.1, 100.0)

        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model.astype(np.float32))
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view.astype(np.float32))
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection.astype(np.float32))

        glBindVertexArray(cube_vao)
        glDrawElements(GL_TRIANGLE_STRIP, len(cube_indices), GL_UNSIGNED_INT, None)
        t += 30 * glfw.get_time()
        glfw.set_time(0)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
