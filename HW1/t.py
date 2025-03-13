import glfw
from OpenGL.GL import *
import numpy as np

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW could not be initialized!")

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(500, 500, "Bresenham's Line Drawing", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window!")

# Set the window's position
glfw.set_window_pos(window, 100, 100)

# Make the window's context current
glfw.make_context_current(window)

# Set up the orthographic projection
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, 500, 0, 500, 0, 1)

# Function to draw a pixel
def draw_pixel(x, y):
    glBegin(GL_POINTS)
    glVertex2i(x, y)
    glEnd()

# Function to draw a line using Bresenham's algorithm
def draw_line(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    incx = 1 if x2 > x1 else -1
    incy = 1 if y2 > y1 else -1
    x, y = x1, y1

    if dx > dy:
        draw_pixel(x, y)
        e = 2 * dy - dx
        inc1 = 2 * (dy - dx)
        inc2 = 2 * dy
        for i in range(dx):
            if e >= 0:
                y += incy
                e += inc1
            else:
                e += inc2
            x += incx
            draw_pixel(x, y)
    else:
        draw_pixel(x, y)
        e = 2 * dx - dy
        inc1 = 2 * (dx - dy)
        inc2 = 2 * dx
        for i in range(dy):
            if e >= 0:
                x += incx
                e += inc1
            else:
                e += inc2
            y += incy
            draw_pixel(x, y)

# Function to display the line
def display(window):
    glClear(GL_COLOR_BUFFER_BIT)
    draw_line(x1, y1, x2, y2)
    glFlush()

# Get user input for the line coordinates
x1, y1, x2, y2 = map(int, input("Enter (x1, y1, x2, y2): ").split())

# Set the display callback
glfw.set_window_refresh_callback(window, display)

# Main loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    display(window)
    glfw.swap_buffers(window)

# Terminate GLFW
glfw.terminate()