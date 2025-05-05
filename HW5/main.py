from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import math

def DrawCircle(radius=1.0, segments=32):
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        glVertex2f(math.cos(angle) * radius, math.sin(angle) * radius)
    glEnd()

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glPushMatrix()
    glPopMatrix()

    glTranslatef(0, 0, 0)

    glPushMatrix()
    glTranslatef(0, 3, 0)
    DrawCircle()
    glPopMatrix()

    glPushMatrix()
    glScalef(1, 2, 1)
    DrawCircle()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(2, 2, 0)
    glRotatef(45, 0, 0, 1)
    glScalef(1.5, 0.5, 1)
    DrawCircle()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(-2, 2, 0)
    glRotatef(-45, 0, 0, 1)
    glScalef(1.5, 0.5, 1)
    DrawCircle()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.5, -3.5, 0)
    glScalef(0.5, 1.5, 1)
    DrawCircle()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(-0.5, -3.5, 0)
    glScalef(0.5, 1.5, 1)
    DrawCircle()
    glPopMatrix()

    glFlush()

if not glfw.init():
    raise Exception("GLFW can't be initialized")

window = glfw.create_window(800, 600, "Lecture 5", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

glfw.make_context_current(window)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluOrtho2D(-10, 10, -10, 10)

while not glfw.window_should_close(window):
    glfw.poll_events()
    display()
    glfw.swap_buffers(window)

glfw.terminate()