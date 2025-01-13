from OpenGL.GL import glGetString, GL_VERSION

try:
    version = glGetString(GL_VERSION).decode()
    print(f"OpenGL version: {version}")
except Exception as e:
    print(f"OpenGL Error: {e}")
