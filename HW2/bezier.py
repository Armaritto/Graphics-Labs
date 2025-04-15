import glfw
from OpenGL.GL import *
import math
if not glfw.init():
    raise Exception("GLFW could not be initialized!")

window_x = 600
window_y = 100

window = glfw.create_window(1000, 800, "Bezier", None, None)

if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window!")

glfw.set_window_pos(window, window_x, window_y)

glfw.make_context_current(window)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, 800, 0, 800, 0, 1)

center = [100,100]
radius = 90
points = [
    [100,10], #0
    [150,10], #1
    [190,50], #2
    [190,100], #3
    [190,100], #4
    [190,150], #5
    [150,190], #6
    [100,190], #7
    [100,190], #8
    [50,190], #9
    [10,150], #10
    [10,100], #11
    [10,100], #12
    [10,50], #13
    [50,10], #14
    [100,10] #15
]

single_curve_pts = [
    points[0],
    points[1],
    points[2],
    points[4],
    points[5],
    points[6],
    points[8],
    points[9],
    points[10],
    points[12], 
    points[13],
    points[14],
    points[15] 
]

def draw_builtin_circle():
    glColor3f(1, 0, 0)
    glBegin(GL_LINE_LOOP)
    for i in range(100):
        angle = 2 * math.pi * i / 100
        x = 100 + 90 * math.cos(angle)
        y = 100 + 90 * math.sin(angle)
        glVertex2f(x, y)
    glEnd()

def cubic_bezier(p0, p1, p2, p3, t):
    return (
        (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0],
        (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
    )

def draw_bezier_single_curve():
    glColor3f(0, 1, 0)
    glLineWidth(2)
    glBegin(GL_LINE_STRIP)
    n = len(single_curve_pts) - 1  # degree = 12
    for step in range(101):
        t = step / 100.0
        x = 0
        y = 0
        for i in range(n + 1):
            b = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            x += b * single_curve_pts[i][0]
            y += b * single_curve_pts[i][1]
        glVertex2f(x, y)
    glEnd()

def draw_control_polygon():
    glColor3f(0.3, 0.3, 0.3)
    glLineWidth(1)
    glBegin(GL_LINE_STRIP)
    for p in points:
        glVertex2f(p[0], p[1])
    glEnd()

    glPointSize(5)
    glBegin(GL_POINTS)
    for p in points:
        glVertex2f(p[0], p[1])
    glEnd()

def draw_bezier_circle():
    glColor3f(0, 0, 1)
    glLineWidth(2)
    glBegin(GL_LINE_STRIP)
    for i in range(0, len(points) - 1, 3):
        p0 = points[i]
        p1 = points[i + 1]
        p2 = points[i + 2]
        p3 = points[i + 3]

        for t in range(0, 101):
            pt = cubic_bezier(p0, p1, p2, p3, t / 100.0)
            glVertex2f(pt[0], pt[1])
    glEnd()

while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT)
    draw_bezier_circle()
    draw_builtin_circle()
    draw_control_polygon()
    draw_bezier_single_curve()
    glFlush()
    glfw.swap_buffers(window)

glfw.terminate()