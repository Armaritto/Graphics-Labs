import glfw
from OpenGL.GL import *
import numpy as np
import trimesh
from pyrr import Matrix44, Vector3
import ctypes
from PIL import Image

################################################################################
# Global Camera Variables
################################################################################
camera_pos = Vector3([0.0, 0.0, 5.0])     # Camera starts behind the model
camera_front = Vector3([0.0, 0.0, -1.0]) # Forward-facing direction
camera_up = Vector3([0.0, 1.0, 0.0])     # Up is +Y

# Angles for mouse look
yaw = -90.0
pitch = 0.0
first_mouse = True

# Last mouse positions
mouse_last_x = 400
mouse_last_y = 300

# Speeds
move_speed = 0.1      # WASD movement
sensitivity = 0.2     # Mouse sensitivity
zoom_speed = 1.0      # Zoom speed

################################################################################
# Shaders
################################################################################
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;       // Vertex position
layout (location = 1) in vec2 aTexCoord;  // Texture coordinate
layout (location = 2) in vec3 aNormal;    // Vertex normal

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 TexCoord;
out vec3 Normal;
out vec3 FragPos;

void main()
{
    // Pass texture coordinate to fragment shader
    TexCoord = aTexCoord;

    // Calculate the fragment position in world space
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = vec3(worldPos);

    // Transform normal to world space
    Normal = mat3(transpose(inverse(model))) * aNormal;

    // Final clip space position
    gl_Position = projection * view * worldPos;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
in vec3 Normal;
in vec3 FragPos;
out vec4 FragColor;

// Texture sampler
uniform sampler2D texture1;

// Lighting uniforms
uniform vec3 lightPos;
uniform vec3 viewPos;      // Camera position
uniform vec3 lightColor;
uniform vec3 objectColor;

void main()
{
    // Sample base color from texture
    vec3 color = texture(texture1, TexCoord).rgb;

    // =============== Ambient ===============
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * lightColor;

    // =============== Diffuse ===============
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // =============== Specular ==============
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    // 32 = shininess factor (adjust as needed)
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    // Combine results
    vec3 lighting = (ambient + diffuse + specular) * color;
    FragColor = vec4(lighting, 1.0);
}
"""

################################################################################
# Model Class
################################################################################
class Model:
    def __init__(self, filepath):
        print("[INFO] Loading model:", filepath)
        # Load mesh via trimesh
        self.mesh = trimesh.load_mesh(filepath, process=False)

        # Center and scale the model for better visibility
        self.center_model()

        # If no texture coords or normals exist, we create zero placeholders
        self.vertices = np.hstack((
            self.mesh.vertices.astype('float32'),
            self.mesh.visual.uv.astype('float32') if self.mesh.visual.uv is not None
            else np.zeros((len(self.mesh.vertices), 2), dtype=np.float32),
            self.mesh.vertex_normals.astype('float32')
        ))

        self.faces = self.mesh.faces.astype('uint32')

        # Load texture from the mesh material if available
        if hasattr(self.mesh.visual.material, "image"):
            image_source = self.mesh.visual.material.image
        else:
            image_source = None

        self.texture_id = self.load_texture(image_source)

        # Create VAO, VBO, and EBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        # Send vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Send face data
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.faces.nbytes, self.faces, GL_STATIC_DRAW)

        stride = 8 * ctypes.sizeof(ctypes.c_float)
        # Positions (location = 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Texture Coordinates (location = 1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # Normals (location = 2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def center_model(self):
        """Center and scale the model to a suitable size in the scene."""
        bounds = self.mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        max_size = np.max(bounds[1] - bounds[0])

        scale_factor = 5.0 / max_size  # Adjust scale
        self.mesh.apply_translation(-center)
        self.mesh.apply_scale(scale_factor)

        print(f"[INFO] Model centered at {center}, scaled by {scale_factor}")

    def load_texture(self, image_source):
        """Load texture from either a file path or a PIL Image object."""
        if image_source is None:
            print("[WARNING] No texture found in material. Using no texture.")
            return None

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            # Already a PIL Image
            img = image_source

        # Convert to RGB, flip top to bottom, then convert to NumPy
        img = img.convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(img, dtype=np.uint8)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        print("[INFO] Texture loaded successfully.")
        return texture

    def render(self):
        # Bind texture if present
        if self.texture_id:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Draw
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.faces.flatten()), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

################################################################################
# Shader Utilities
################################################################################
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if status != GL_TRUE:
        error_log = glGetShaderInfoLog(shader)
        raise RuntimeError(f"Shader compile error: {error_log.decode('utf-8')}")
    return shader

def create_shader_program():
    vertex_shader = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    # Check link status
    link_status = glGetProgramiv(program, GL_LINK_STATUS)
    if link_status != GL_TRUE:
        error_log = glGetProgramInfoLog(program)
        raise RuntimeError(f"Program link error: {error_log.decode('utf-8')}")

    # Cleanup
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program

################################################################################
# Camera Control Callbacks
################################################################################
def mouse_callback(window, xpos, ypos):
    global yaw, pitch, mouse_last_x, mouse_last_y, first_mouse, camera_front

    if first_mouse:
        mouse_last_x = xpos
        mouse_last_y = ypos
        first_mouse = False

    x_offset = (xpos - mouse_last_x) * sensitivity
    y_offset = (mouse_last_y - ypos) * sensitivity
    mouse_last_x, mouse_last_y = xpos, ypos

    yaw += x_offset
    pitch += y_offset

    # Clamp pitch to avoid flipping
    if pitch > 89.0:
        pitch = 89.0
    if pitch < -89.0:
        pitch = -89.0

    # Calculate new front vector
    direction = Vector3([
        np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
        np.sin(np.radians(pitch)),
        np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
    ])
    camera_front = direction.normalized

def scroll_callback(window, xoffset, yoffset):
    global camera_pos, camera_front
    # Zoom in/out by moving camera along its front vector
    zoom_factor = yoffset * zoom_speed
    camera_pos += camera_front * zoom_factor

def key_callback(window, key, scancode, action, mods):
    global camera_pos, camera_front, camera_up

    if action == glfw.PRESS or action == glfw.REPEAT:
        # Basic WASD + Arrow keys
        if key == glfw.KEY_W or key == glfw.KEY_UP:
            camera_pos += camera_front * move_speed
        if key == glfw.KEY_S or key == glfw.KEY_DOWN:
            camera_pos -= camera_front * move_speed
        if key == glfw.KEY_A or key == glfw.KEY_LEFT:
            left = camera_front.cross(camera_up).normalized
            camera_pos -= left * move_speed
        if key == glfw.KEY_D or key == glfw.KEY_RIGHT:
            right = camera_up.cross(camera_front).normalized
            camera_pos -= right * move_speed
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

################################################################################
# Main Application
################################################################################
def main():
    # Initialize GLFW
    if not glfw.init():
        print("[ERROR] Could not initialize GLFW")
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "OBJ Viewer with Phong Lighting", None, None)
    if not window:
        glfw.terminate()
        print("[ERROR] Could not create window")
        return

    glfw.make_context_current(window)
    # Register callbacks
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    # Configure some OpenGL state
    glEnable(GL_DEPTH_TEST)

    # Load Model (replace with your OBJ)
    model = Model("assets/models/bugatti.obj")

    # Create shader program
    shader_program = create_shader_program()

    # Lighting & Colors
    light_pos = Vector3([5.0, 5.0, 5.0])
    light_color = Vector3([1.0, 1.0, 1.0])
    object_color = Vector3([1.0, 1.0, 1.0])  # If you want to tint your texture

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)

        # Create projection matrix
        aspect_ratio = 800 / 600
        projection = Matrix44.perspective_projection(45.0, aspect_ratio, 0.1, 100.0)

        # Create view matrix
        target = camera_pos + camera_front
        view = Matrix44.look_at(camera_pos, target, camera_up)

        # Set uniform matrices
        view_loc = glGetUniformLocation(shader_program, "view")
        proj_loc = glGetUniformLocation(shader_program, "projection")
        model_loc = glGetUniformLocation(shader_program, "model")
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, Matrix44.identity())

        # Set lighting uniforms
        light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
        view_pos_loc = glGetUniformLocation(shader_program, "viewPos")
        light_color_loc = glGetUniformLocation(shader_program, "lightColor")
        obj_color_loc = glGetUniformLocation(shader_program, "objectColor")

        glUniform3fv(light_pos_loc, 1, light_pos)
        glUniform3fv(view_pos_loc, 1, camera_pos)
        glUniform3fv(light_color_loc, 1, light_color)
        glUniform3fv(obj_color_loc, 1, object_color)

        # Render the loaded model
        model.render()

        # Error checking
        err = glGetError()
        if err != GL_NO_ERROR:
            print(f"[ERROR] OpenGL Error: {err}")

        # Swap buffers
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Cleanup
    glfw.terminate()

if __name__ == "__main__":
    main()
