import glfw
from OpenGL.GL import *
import numpy as np

if not glfw.init():
    raise Exception("GLFW could not be initialized!")

window_x = 600
window_y = 100

window = glfw.create_window(1000, 800, "Armia", None, None)

if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window!")

glfw.set_window_pos(window, window_x, window_y)

glfw.make_context_current(window)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, 800, 0, 800, 0, 1)

def drawline(v1, v2):
    v1 = [int(v1[0] * 50), int(v1[1] * 50)]
    v2 = [int(v2[0] * 50), int(v2[1] * 50)]
    dx = abs(v2[0] - v1[0])
    dy = abs(v2[1] - v1[1])
    incx = 1 if v2[0] > v1[0] else -1
    incy = 1 if v2[1] > v1[1] else -1
    x, y = v1[0], v1[1]

    if dx > dy:
        glBegin(GL_POINTS)
        glVertex2i(x, y)
        glEnd()
        e = 2 * dy - dx
        inc1 = 2 * (dy - dx)
        inc2 = 2 * dy
        for _ in range(dx):
            if e >= 0:
                y += incy
                e += inc1
            else:
                e += inc2
            x += incx
            glBegin(GL_POINTS)
            glVertex2i(x, y)
            glEnd()
    else:
        glBegin(GL_POINTS)
        glVertex2i(x, y)
        glEnd()
        e = 2 * dx - dy
        inc1 = 2 * (dx - dy)
        inc2 = 2 * dx
        for _ in range(dy):
            if e >= 0:
                x += incx
                e += inc1
            else:
                e += inc2
            y += incy
            glBegin(GL_POINTS)
            glVertex2i(x, y)
            glEnd()
    
a = [1.0,1.0,0.0]
b = [2.0,4.0,0.0]
c = [3.0,1.0,0.0]
d = [1.5,2.5,0,0]
e = [2.6,2.2,0.0]

f = [3.78,1.05,0,0]
g = [4.0,4.0,0.0]
h = [4.67,3.0,0.0]
i = [3.90,2.68,0.0]
j = [5.0,1.0,0.0]

k = [5.5,1.0,0.0]
l = [6.0,4.0,0.0]
m = [6.5,3.0,0.0]
n = [7.0,4.0,0.0]
o = [7.5,1.0,0.0]

p = [7.5,3.83,0.0]
q = [9.5,4.16,0.0]
r = [8.0,0.83,0.0]
s = [10.0,1.16,0.0]
t = [8.5,4.0,0.0]
u = [9.0,1.0,0.0]

v = [10.5,1.0,0.0]
w = [11.5,4.0,0.0]
x = [12.5,1.0,0.0]
y = [11.0,2.5,0.0]
z = [12.1,2.2,0.0]


while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT)

    # Draw A
    drawline(a, b)
    drawline(b, c)
    drawline(d, e)
    
    # Draw R
    drawline(f, g)
    drawline(g, h)
    drawline(h, i)
    drawline(i, j)

    # Draw M
    drawline(k, l)
    drawline(l, m)
    drawline(m, n)
    drawline(n, o)

    # Draw I
    drawline(p, q)
    drawline(r, s)
    drawline(t, u)

    # Draw A
    drawline(v, w)
    drawline(w, x)
    drawline(y, z)

    glFlush()
    glfw.swap_buffers(window)

glfw.terminate()