"""
OpenGL Cloth and Ball Physics Simulation
Computer Graphics Project - COMP 342

MODIFIED VERSION
----------------
- The cloth is now stationary, pinned along all FOUR sides to a rigid
  frame (like a trampoline), instead of falling freely.
- The cloth's orientation can be set at construction time (rx, ry, rz
  in degrees) and adjusted live with I/K and J/L.
- The ball now has real physics (gravity, velocity, floor collision)
  and FALLS onto the cloth, rather than sitting static while the cloth
  falls onto it.
- Collision is now two-way: the cloth still deforms exactly as before
  (original particle/spring solver untouched), but it also pushes back
  on the ball (Newton's third law), so the ball decelerates and settles
  into the cloth like a real object landing on a trampoline.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

class Particle:
    def __init__(self, x, y, z, mass=1.0):
        self.pos = np.array([x, y, z], dtype=float)
        self.old_pos = np.array([x, y, z], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.force = np.array([0.0, 0.0, 0.0], dtype=float)
        self.mass = mass
        self.pinned = False
        self.normal = np.array([0.0, 0.0, 1.0], dtype=float)
    
    def apply_force(self, f):
        if not self.pinned:
            self.force += f
    
    def update(self, dt):
        if self.pinned:
            return
        
        acceleration = self.force / self.mass
        new_pos = 2 * self.pos - self.old_pos + acceleration * dt * dt
        self.velocity = (new_pos - self.pos) / dt
        self.old_pos = self.pos.copy()
        self.pos = new_pos
        self.force = np.array([0.0, 0.0, 0.0], dtype=float)
    
    def add_damping(self, damping=0.99):
        if not self.pinned:
            self.velocity *= damping

class Spring:
    def __init__(self, p1, p2, stiffness=50.0):
        self.p1 = p1
        self.p2 = p2
        self.rest_length = np.linalg.norm(p1.pos - p2.pos)
        self.stiffness = stiffness
    
    def satisfy_constraint(self):
        delta = self.p2.pos - self.p1.pos
        current_length = np.linalg.norm(delta)
        
        if current_length == 0:
            return
        
        correction = (current_length - self.rest_length) / current_length
        correction_vector = delta * correction * 0.8
        
        if not self.p1.pinned:
            self.p1.pos += correction_vector
        if not self.p2.pinned:
            self.p2.pos -= correction_vector

class Cloth:
    def __init__(self, width, height, resolution, start_pos=(0, 3, 0),
                 orientation=(0, 0, 0), pin_edges=True):
        """
        orientation: (rx, ry, rz) rotation in degrees applied to the cloth
                     mesh about its own center before it is placed at
                     start_pos. (0, 0, 0) keeps the cloth flat/horizontal,
                     matching the original behavior.
        pin_edges:   if True, every particle along all four border
                     rows/columns is pinned in place, simulating a cloth
                     stretched over a rigid frame (like a trampoline).
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.start_pos = start_pos
        self.orientation = orientation
        self.pin_edges = pin_edges
        self.particles = []
        self.springs = []
        
        self.create_mesh()
        if self.pin_edges:
            self.pin_border()
        self.create_springs()
    
    def _rotation_matrix(self):
        rx, ry, rz = [math.radians(a) for a in self.orientation]
        
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])
        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx
    
    def create_mesh(self):
        step_x = self.width / (self.resolution - 1)
        step_y = self.height / (self.resolution - 1)
        R = self._rotation_matrix()
        
        for i in range(self.resolution):
            row = []
            for j in range(self.resolution):
                # Build the point in local (flat) space, centered on origin,
                # then rotate it into the requested orientation.
                local_x = -self.width / 2 + j * step_x
                local_y = 0.0
                local_z = -self.height / 2 + i * step_y
                local = np.array([local_x, local_y, local_z])
                
                rotated = R @ local
                x = self.start_pos[0] + rotated[0]
                y = self.start_pos[1] + rotated[1]
                z = self.start_pos[2] + rotated[2]
                
                particle = Particle(x, y, z, mass=0.1)
                row.append(particle)
            self.particles.append(row)
    
    def pin_border(self):
        """Attach all four sides of the cloth to a rigid frame."""
        last_i = self.resolution - 1
        last_j = self.resolution - 1
        for i in range(self.resolution):
            for j in range(self.resolution):
                if i == 0 or i == last_i or j == 0 or j == last_j:
                    self.particles[i][j].pinned = True
    
    def create_springs(self):
        for i in range(self.resolution):
            for j in range(self.resolution):
                if j < self.resolution - 1:
                    spring = Spring(self.particles[i][j], self.particles[i][j+1])
                    self.springs.append(spring)
                
                if i < self.resolution - 1:
                    spring = Spring(self.particles[i][j], self.particles[i+1][j])
                    self.springs.append(spring)
        
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                spring1 = Spring(self.particles[i][j], self.particles[i+1][j+1])
                spring2 = Spring(self.particles[i][j+1], self.particles[i+1][j])
                self.springs.append(spring1)
                self.springs.append(spring2)
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                if j < self.resolution - 2:
                    spring = Spring(self.particles[i][j], self.particles[i][j+2], stiffness=30.0)
                    self.springs.append(spring)
                
                if i < self.resolution - 2:
                    spring = Spring(self.particles[i][j], self.particles[i+2][j], stiffness=30.0)
                    self.springs.append(spring)
    
    def apply_gravity(self, gravity=(0, -9.8, 0)):
        g = np.array(gravity, dtype=float)
        for row in self.particles:
            for particle in row:
                particle.apply_force(g * particle.mass)
    
    def apply_wind(self, wind_force=(0, 0, 0)):
        w = np.array(wind_force, dtype=float)
        for row in self.particles:
            for particle in row:
                particle.apply_force(w)
    
    def update(self, dt, constraint_iterations=4):
        self.apply_gravity()
        
        for row in self.particles:
            for particle in row:
                particle.update(dt)
                particle.add_damping(0.99)
        
        for _ in range(constraint_iterations):
            for spring in self.springs:
                spring.satisfy_constraint()
    
    def compute_normals(self):  # Vertex normals
        for row in self.particles:
            for particle in row:
                particle.normal = np.array([0.0, 0.0, 0.0], dtype=float)
        
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                p1 = self.particles[i][j]
                p2 = self.particles[i][j+1]
                p3 = self.particles[i+1][j]
                p4 = self.particles[i+1][j+1]
                
                v1 = p2.pos - p1.pos
                v2 = p3.pos - p1.pos
                normal1 = np.cross(v1, v2)
                if np.linalg.norm(normal1) > 0:
                    normal1 = normal1 / np.linalg.norm(normal1)
                
                v3 = p3.pos - p2.pos
                v4 = p4.pos - p2.pos
                normal2 = np.cross(v3, v4)
                if np.linalg.norm(normal2) > 0:
                    normal2 = normal2 / np.linalg.norm(normal2)
                
                p1.normal += normal1
                p2.normal += normal1 + normal2
                p3.normal += normal1 + normal2
                p4.normal += normal2
        
        for row in self.particles:
            for particle in row:
                norm = np.linalg.norm(particle.normal)
                if norm > 0:
                    particle.normal /= norm
    
    def collide_with_sphere(self, ball, dt, contact_stiffness=400.0):
        """
        Resolve collision between the cloth and a MOVING ball.

        The cloth particles are pushed out of the sphere exactly as in
        the original code. In addition, the cloth now pushes back on the
        ball (Newton's third law): the deeper the cloth is pressed into
        the ball, the harder it pushes back, so the ball realistically
        decelerates, sinks in, and settles/bounces depending on speed.
        """
        center = ball.position
        radius = ball.radius
        collision_radius = radius + 0.02
        
        reaction_impulse = np.array([0.0, 0.0, 0.0])
        contacts = 0
        
        for row in self.particles:
            for particle in row:
                delta = particle.pos - center
                distance = np.linalg.norm(delta)
                
                if distance < collision_radius:
                    if distance > 0.001:
                        normal = delta / distance
                    else:
                        normal = np.array([0.0, 1.0, 0.0])
                    
                    penetration = collision_radius - distance
                    
                    if not particle.pinned:
                        particle.pos = center + normal * collision_radius
                        velocity_normal = np.dot(particle.velocity, normal)
                        if velocity_normal < 0:
                            particle.velocity -= velocity_normal * normal * 1.5
                        particle.velocity *= 0.5
                    
                    # Soft-contact reaction pushing back on the ball,
                    # proportional to how far the cloth is pressed in.
                    reaction_impulse += -normal * penetration * contact_stiffness * dt
                    contacts += 1
        
        if contacts > 0:
            avg_impulse = reaction_impulse / contacts
            ball.velocity += avg_impulse / ball.mass
    
    def render(self, wireframe=False):  # OpenGL rendering
        self.compute_normals()
        
        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
            glColor3f(0.0, 0.0, 0.0)
            glLineWidth(1.5)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBegin(GL_QUADS)  # Quad mesh
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                p1 = self.particles[i][j]
                p2 = self.particles[i][j+1]
                p3 = self.particles[i+1][j+1]
                p4 = self.particles[i+1][j]
                
                if not wireframe:
                    glColor3f(1.0, 1.0, 1.0)  # White cloth
                
                glNormal3fv(p1.normal)  # Phong shading
                glVertex3fv(p1.pos)
                glNormal3fv(p2.normal)
                glVertex3fv(p2.pos)
                glNormal3fv(p3.normal)
                glVertex3fv(p3.pos)
                glNormal3fv(p4.normal)
                glVertex3fv(p4.pos)
        glEnd()
        
        if wireframe:
            glEnable(GL_LIGHTING)
            glLineWidth(1.0)
        
        if not wireframe:
            glDisable(GL_LIGHTING)
            glColor3f(0.0, 0.0, 0.0)  # Black grid
            glLineWidth(1.0)
            
            glBegin(GL_LINES)
            for i in range(self.resolution):
                for j in range(self.resolution - 1):
                    p1 = self.particles[i][j]
                    p2 = self.particles[i][j+1]
                    glVertex3fv(p1.pos)
                    glVertex3fv(p2.pos)
            glEnd()
            
            glBegin(GL_LINES)
            for i in range(self.resolution - 1):
                for j in range(self.resolution):
                    p1 = self.particles[i][j]
                    p2 = self.particles[i+1][j]
                    glVertex3fv(p1.pos)
                    glVertex3fv(p2.pos)
            glEnd()
            
            glEnable(GL_LIGHTING)
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

class Ball:
    def __init__(self, position, radius, color=(1.0, 0.3, 0.3), mass=5.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.radius = radius
        self.color = color
        self.mass = mass
    
    def update(self, dt, gravity=(0, -9.8, 0), floor_y=-1.9, damping=0.995):
        g = np.array(gravity, dtype=float)
        self.velocity += g * dt
        self.velocity *= damping
        self.position += self.velocity * dt
        
        # Floor collision so the ball can't fall forever if it slips
        # past the cloth's edges.
        if self.position[1] - self.radius < floor_y:
            self.position[1] = floor_y + self.radius
            if self.velocity[1] < 0:
                self.velocity[1] *= -0.3
    
    def render(self):  # GLU sphere
        glPushMatrix()
        glTranslatef(*self.position)
        glColor3f(*self.color)
        
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        gluSphere(quad, self.radius, 32, 32)
        gluDeleteQuadric(quad)
        
        glPopMatrix()

class Button:
    def __init__(self, x, y, width, height, text, value):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.value = value
        self.hovered = False
        self.active = False
    
    def check_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
        return self.hovered
    
    def check_click(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)
    
    def draw(self, screen, font):
        if self.active:
            color = (100, 200, 100)
        elif self.hovered:
            color = (80, 80, 120)
        else:
            color = (50, 50, 80)
        
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)
        
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

class ClothSimulation:
    def __init__(self, cloth_resolution=12, cloth_orientation=(0, 0, 0)):
        pygame.init()
        self.display = (1200, 800)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption('Cloth Drop Simulation - Interactive')
        
        self.ui_surface = pygame.Surface(self.display, pygame.SRCALPHA)
        
        self.setup_opengl()
        
        # The cloth is now a stationary trampoline; the ball falls onto it.
        self.cloth_height = 3.0
        self.drop_height = 8.0
        
        self.cloth = Cloth(width=4, height=4, resolution=cloth_resolution,
                            start_pos=(0, self.cloth_height, 0),
                            orientation=cloth_orientation, pin_edges=True)
        self.ball = Ball(position=(0, self.drop_height, 0), radius=1.0)
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.wireframe = False
        self.rotation = [20, 30]
        self.zoom = -14
        
        self.mouse_down = False
        self.last_mouse_pos = None
        self.ui_active = False
        
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.buttons = []
        button_width = 70
        button_height = 40
        button_spacing = 10
        start_x = 20
        start_y = 20
        
        resolutions = [
            (4, "4x4"), (6, "6x6"), (8, "8x8"), (10, "10x10"),
            (12, "12x12"), (14, "14x14"), (16, "16x16"), (32, "32x32")
        ]
        
        for i, (res, label) in enumerate(resolutions):
            row = i // 4
            col = i % 4
            button = Button(
                start_x + col * (button_width + button_spacing),
                start_y + row * (button_height + button_spacing),
                button_width,
                button_height,
                label,
                res
            )
            if res == cloth_resolution:
                button.active = True
            self.buttons.append(button)
        
        self.reset_button = Button(
            start_x,
            start_y + 2 * (button_height + button_spacing) + 10,
            180,
            button_height,
            "RESET",
            None
        )
    
    def setup_opengl(self):  # Lighting setup
        glEnable(GL_DEPTH_TEST)  # Z-buffer
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLight(GL_LIGHT0, GL_POSITION, (5, 10, 5, 1))
        glLight(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLight(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        glLight(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))
        
        glMaterial(GL_FRONT, GL_SPECULAR, (1, 1, 1, 1))
        glMaterial(GL_FRONT, GL_SHININESS, 50)
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)  # Perspective
        glMatrixMode(GL_MODELVIEW)
    
    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        
        self.ui_active = False
        for button in self.buttons:
            if button.check_hover(mouse_pos):
                self.ui_active = True
        if self.reset_button.check_hover(mouse_pos):
            self.ui_active = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_w:
                    self.wireframe = not self.wireframe
                elif event.key == pygame.K_r:
                    self.reset_cloth()
                elif event.key == pygame.K_1:
                    self.change_resolution(4)
                elif event.key == pygame.K_2:
                    self.change_resolution(6)
                elif event.key == pygame.K_3:
                    self.change_resolution(8)
                elif event.key == pygame.K_4:
                    self.change_resolution(10)
                elif event.key == pygame.K_5:
                    self.change_resolution(12)
                elif event.key == pygame.K_6:
                    self.change_resolution(14)
                elif event.key == pygame.K_7:
                    self.change_resolution(16)
                elif event.key == pygame.K_8:
                    self.change_resolution(32)
                elif event.key == pygame.K_i:
                    self.adjust_orientation(rx_delta=10)
                elif event.key == pygame.K_k:
                    self.adjust_orientation(rx_delta=-10)
                elif event.key == pygame.K_j:
                    self.adjust_orientation(rz_delta=-10)
                elif event.key == pygame.K_l:
                    self.adjust_orientation(rz_delta=10)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    clicked_button = False
                    for button in self.buttons:
                        if button.check_click(mouse_pos):
                            self.change_resolution(button.value)
                            clicked_button = True
                            break
                    
                    if self.reset_button.check_click(mouse_pos):
                        self.reset_cloth()
                        clicked_button = True
                    
                    if not clicked_button and not self.ui_active:
                        self.mouse_down = True
                        self.last_mouse_pos = mouse_pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_down and not self.ui_active:
                    if self.last_mouse_pos:
                        dx = mouse_pos[0] - self.last_mouse_pos[0]
                        dy = mouse_pos[1] - self.last_mouse_pos[1]
                        self.rotation[0] += dy * 0.5
                        self.rotation[1] += dx * 0.5
                    self.last_mouse_pos = mouse_pos
            
            elif event.type == pygame.MOUSEWHEEL:
                self.zoom += event.y * 0.5
                self.zoom = max(-30, min(-5, self.zoom))
    
    def change_resolution(self, new_resolution):
        orientation = self.cloth.orientation
        self.cloth = Cloth(width=4, height=4, resolution=new_resolution,
                            start_pos=(0, self.cloth_height, 0),
                            orientation=orientation, pin_edges=True)
        self.reset_ball()
        for button in self.buttons:
            button.active = (button.value == new_resolution)
    
    def reset_cloth(self):
        current_res = self.cloth.resolution
        orientation = self.cloth.orientation
        self.cloth = Cloth(width=4, height=4, resolution=current_res,
                            start_pos=(0, self.cloth_height, 0),
                            orientation=orientation, pin_edges=True)
        self.reset_ball()
    
    def adjust_orientation(self, rx_delta=0, ry_delta=0, rz_delta=0):
        """Live-tilt the cloth and re-drop the ball onto the new orientation."""
        rx, ry, rz = self.cloth.orientation
        new_orientation = (rx + rx_delta, ry + ry_delta, rz + rz_delta)
        current_res = self.cloth.resolution
        self.cloth = Cloth(width=4, height=4, resolution=current_res,
                            start_pos=(0, self.cloth_height, 0),
                            orientation=new_orientation, pin_edges=True)
        self.reset_ball()
    
    def reset_ball(self):
        self.ball.position = np.array([0, self.drop_height, 0], dtype=float)
        self.ball.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glTranslatef(0, -1, self.zoom)  # Camera transform
        glRotatef(self.rotation[0], 1, 0, 0)  # Rotation
        glRotatef(self.rotation[1], 0, 1, 0)
        
        self.draw_grid()
        self.ball.render()
        self.cloth.render(wireframe=self.wireframe)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)  # 2D UI
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        self.render_ui_opengl()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        pygame.display.flip()
    
    def draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINES)
        for i in range(-10, 11):
            glVertex3f(i, -2, -10)
            glVertex3f(i, -2, 10)
            glVertex3f(-10, -2, i)
            glVertex3f(10, -2, i)
        glEnd()
        glEnable(GL_LIGHTING)
    
    def render_ui_opengl(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glColor4f(0.0, 0.0, 0.0, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(360, 10)
        glVertex2f(360, 160)
        glVertex2f(10, 160)
        glEnd()
        
        for button in self.buttons:
            self.draw_button_opengl(button)
        self.draw_button_opengl(self.reset_button)
        
        self.draw_text_overlay()
        
        glDisable(GL_BLEND)
    
    def draw_button_opengl(self, button):
        if button.active:
            color = (0.4, 0.8, 0.4, 0.9)
        elif button.hovered:
            color = (0.35, 0.35, 0.5, 0.9)
        else:
            color = (0.2, 0.2, 0.35, 0.9)
        
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(button.rect.left, button.rect.top)
        glVertex2f(button.rect.right, button.rect.top)
        glVertex2f(button.rect.right, button.rect.bottom)
        glVertex2f(button.rect.left, button.rect.bottom)
        glEnd()
        
        glColor4f(0.8, 0.8, 0.8, 1.0)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        glVertex2f(button.rect.left, button.rect.top)
        glVertex2f(button.rect.right, button.rect.top)
        glVertex2f(button.rect.right, button.rect.bottom)
        glVertex2f(button.rect.left, button.rect.bottom)
        glEnd()
    
    def draw_text_overlay(self):
        text_surface = pygame.Surface(self.display, pygame.SRCALPHA)
        text_surface.fill((0, 0, 0, 0))
        
        button_font = pygame.font.Font(None, 30)
        for button in self.buttons:
            text_surf = button_font.render(button.text, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=button.rect.center)
            shadow_surf = button_font.render(button.text, True, (0, 0, 0))
            shadow_rect = shadow_surf.get_rect(center=(button.rect.centerx + 1, button.rect.centery + 1))
            text_surface.blit(shadow_surf, shadow_rect)
            text_surface.blit(text_surf, text_rect)
        
        reset_font = pygame.font.Font(None, 32)
        text_surf = reset_font.render(self.reset_button.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.reset_button.rect.center)
        shadow_surf = reset_font.render(self.reset_button.text, True, (0, 0, 0))
        shadow_rect = shadow_surf.get_rect(center=(self.reset_button.rect.centerx + 1, self.reset_button.rect.centery + 1))
        text_surface.blit(shadow_surf, shadow_rect)
        text_surface.blit(text_surf, text_rect)
        
        rx, ry, rz = self.cloth.orientation
        info_texts = [
            f"Particles: {self.cloth.resolution * self.cloth.resolution}",
            f"Springs: {len(self.cloth.springs)}",
            f"Orientation (rx, ry, rz): ({rx:.0f}, {ry:.0f}, {rz:.0f})",
            "Keys: SPACE=Pause | W=Wireframe | R=Reset | 1-8=Resolution | I/K,J/L=Tilt"
        ]
        
        y_offset = self.display[1] - 95
        for text in info_texts:
            shadow = self.small_font.render(text, True, (0, 0, 0))
            text_surface.blit(shadow, (21, y_offset + 1))
            surface = self.small_font.render(text, True, (255, 255, 255))
            text_surface.blit(surface, (20, y_offset))
            y_offset += 23
        
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = (255, 100, 100) if self.paused else (100, 255, 100)
        status_font = pygame.font.Font(None, 36)
        shadow = status_font.render(status, True, (0, 0, 0))
        text_surface.blit(shadow, (self.display[0] - 121, 21))
        surface = status_font.render(status, True, status_color)
        text_surface.blit(surface, (self.display[0] - 120, 20))
        
        texture_data = pygame.image.tostring(text_surface, "RGBA", True)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glRasterPos2f(0, 0)
        glDrawPixels(self.display[0], self.display[1], GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glDisable(GL_BLEND)
    
    def update(self, dt):
        if not self.paused:
            substeps = 5
            sub_dt = dt / substeps
            
            for _ in range(substeps):
                self.ball.update(sub_dt)
                self.cloth.update(sub_dt)
                self.cloth.collide_with_sphere(self.ball, sub_dt)
    
    def run(self):
        while self.running:
            dt = min(self.clock.tick(60) / 1000.0, 0.02)
            
            self.handle_events()
            self.update(dt)
            self.render()
        
        pygame.quit()

def main():
    print("=" * 60)
    print("CLOTH TRAMPOLINE SIMULATION")
    print("Computer Graphics Project")
    print("=" * 60)
    print("\nA ball now falls onto a cloth pinned along all four sides.")
    print("\nControls:")
    print("- Resolution Buttons: Click buttons for different resolutions")
    print("  4x4, 6x6, 8x8, 10x10, 12x12, 14x14, 16x16, 32x32")
    print("- Keyboard: Press 1-8 for quick resolution change")
    print("- RESET: Click RESET button or press R to re-drop the ball")
    print("- SPACE: Pause/Resume simulation")
    print("- W: Toggle Wireframe mode")
    print("- I/K: Tilt cloth around X axis")
    print("- J/L: Tilt cloth around Z axis")
    print("- Mouse Drag: Rotate camera view")
    print("- Mouse Scroll: Zoom in/out")
    print("- ESC: Exit simulation")
    print("\nWatch the ball fall onto the trampoline-style cloth!")
    print("=" * 60)
    
    sim = ClothSimulation(cloth_resolution=12)
    sim.run()

if __name__ == "__main__":
    main()
