#Ray Tracing algorithm
#Variables
#Class Ray with Origin and Direction
#Class Sphere Object to test Intersection 
#Color RGB Values

#Function Ray-Sphere Intersection
#Lighting  Phong shading
#Render Loop
#Ray from camera
#Intersection with objectes
#Shade based on hit or background
import numpy as np
from PIL import Image

class Ray:
    def __init__(self, origin, direction):
        # Camera or viewer origin point 
        self.origin = origin
        #Direction vector of the ray point
        self.direction = direction/np.linalg.norm(direction)
    def point_at_parameter(self, t):
        #Calculate point along the ray at a given parameter
        return self.origin + t * self.direction

import numpy as np

class Sphere:
    def __init__(self, center, radius, resolution=360):
        self.center = np.array(center, dtype=float)
        self.radius = radius

        # Generación de coordenadas esféricas solo si quieres visualización
        phi = np.linspace(0, 2*np.pi, 2*resolution)
        theta = np.linspace(0, np.pi, resolution)
        self.phi, self.theta = np.meshgrid(phi, theta)

    def get_surface_points(self):
        r_xy = self.radius * np.sin(self.theta)
        x = self.center[0] + np.cos(self.phi) * r_xy
        y = self.center[1] + np.sin(self.phi) * r_xy
        z = self.center[2] + self.radius * np.cos(self.theta)
        return np.stack([x, y, z])

    def intersect(self, ray):
        origin_to_center = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(origin_to_center, ray.direction)
        c = np.dot(origin_to_center, origin_to_center) - self.radius ** 2
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None
        else:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            # Escoge la intersección más cercana en frente de la cámara
            t = t1 if t1 > 0 else (t2 if t2 > 0 else None)
            if t is not None:
                return ray.point_at_parameter(t)
            return None

#Shading (cambiar)
def phong_shading(hit_point, normal, view_dir, light_pos, light_color, object_color):
    light_dir = (light_pos - hit_point)
    light_dir /= np.linalg.norm(light_dir)
    normal = normal / np.linalg.norm(normal)
    view_dir = view_dir/np.linalg.norm(view_dir)

    #componentes
    ambient_component =ambient * object_color
    diff = max(np.dot(normal, light_dir), 0.0)
    diffuse_component = diffuse_coef * diff * object_color
    
    reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
    spec = max(np.dot(view_dir, reflect_dir), 0.0) ** shininess
    specular_component = specular_coeff * spec * light_color

    color = ambient_component + diffuse_component + specular_component
    return np.clip(color, 0, 255).astype(np.uint8)

#Visualizacion
width, height = 400, 400
aspect_ratio = width/height
fov = np.pi /3
imagen = Image.new("RGB", (width, height))
pixels =imagen.load()

#Ray
camera_origin =np.array([0,0,0],dtype=float)

#Esfera
sphere_center=np.array([0,0,-3],dtype=float)
radius_sphere= 1.0
esfera=Sphere(sphere_center, radius_sphere)

#Iluminacion
light_position = np.array([5,5,0])
light_color = np.array([255, 255, 255])
ambient = 0.1
diffuse_coef = 0.6
specular_coeff = 0.3
shininess = 50

sphere_color = np.array([255, 0, 0], dtype=float)
background_color = (135,206,250)

for i in range(width):
    for j in range(height):
        x = (2 * (i + 0.5) / width - 1) * np.tan(fov / 2) * aspect_ratio
        y = (1 - 2 * (j + 0.5) / height) * np.tan(fov / 2)
        direction = np.array([x, y, -1], dtype=float)
        ray = Ray(camera_origin, direction)
        hit_point = esfera.intersect(ray)
        if hit_point is not None:
            normal = hit_point - sphere_center
            view_dir = - ray.direction
            
            color = phong_shading(hit_point, normal, view_dir, light_position, light_color, sphere_color)
            pixels[i, j] = tuple(color)
        else:
            pixels[i, j] = background_color


imagen.show()