import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import math

R = 1.0 
r = 0.4  
n_u = 40
n_v = 20

def compute_vertex(u, v):
    theta = 2 * math.pi * u
    phi = 2 * math.pi * v
    x = (R + r * math.cos(phi)) * math.cos(theta)
    y = (R + r * math.cos(phi)) * math.sin(theta)
    z = r * math.sin(phi)
    return x, y, z

def draw_torus_wireframe():
    du = 1.0 / n_u
    dv = 1.0 / n_v
    glColor3f(1, 1, 1)
    for i in range(n_u):
        u = i * du
        next_u = ((i + 1) % n_u) * du
        glBegin(GL_LINE_STRIP)
        for j in range(n_v + 1):
            v = j * dv
            glVertex3fv(compute_vertex(u, v))
            glVertex3fv(compute_vertex(next_u, v))
        glEnd()


if not glfw.init():
    raise Exception("Could not initialize GLFW")

window = glfw.create_window(768, 768, "ArmaTorus", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")

glfw.set_window_pos(window, 100, 100)
glfw.make_context_current(window)
glEnable(GL_DEPTH_TEST)

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, 1, 0.1, 100)

glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
gluLookAt(3, 3, 3, 0, 0, 0, 0, 0, 1) 

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glPushMatrix()
    draw_torus_wireframe()
    glPopMatrix()

    glfw.swap_buffers(window)

glfw.terminate()
