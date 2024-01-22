from PIL import Image
from functools import reduce
import numpy as np
import time

# Vector logic
class Vector3D():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other):
        return Vector3D(self.x * other, self.y * other, self.z * other)

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self):
        return self.dot(self)

    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def components(self):
        return (self.x, self.y, self.z)

Color = Vector3D

# Resolution
(width, height) = (400, 400)
LightPosition = Vector3D(5, 5., -10)
CameraPosition = Vector3D(0., 0.35, -1.)
FAR_AWAY = 1.0e39  # vädligt stort avstånd

def raytrace(ray_origin, ray_direction, scene, bounce=0):
    # ray_origin is the ray origin, ray_direction is the normalized ray direction
    # scene is a list of Sphere objects
    # bounce is the number of the bounce, starting at zero for camera rays

    distances = [sphere.intersect(ray_origin, ray_direction) for sphere in scene]
    nearest = reduce(np.minimum, distances)
    color = Color(0, 0, 0)
    
    for (sphere, distance) in zip(scene, distances):
        color += sphere.light(ray_origin, ray_direction, distance, scene, bounce) * (nearest != FAR_AWAY) * (distance == nearest)

    return color

class Sphere:
    def __init__(self, center, radius, diffuse_color, mirror=0.7):
        self.center = center
        self.radius = radius
        self.diffuse_color = diffuse_color
        self.mirror = mirror

    def intersect(self, ray_origin, ray_direction):
        b = 2 * ray_direction.dot(ray_origin - self.center)
        c = abs(self.center) + abs(ray_origin) - 2 * self.center.dot(ray_origin) - (self.radius * self.radius)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)

        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FAR_AWAY)

    def get_diffuse_color(self, intersection_point):
        return self.diffuse_color

    def light(self, ray_origin, ray_direction, distance, scene, bounce):
        intersection_point = (ray_origin + ray_direction * distance)
        normal = (intersection_point - self.center) * (1. / self.radius)
        to_light = (LightPosition - intersection_point).norm()
        to_observer = (CameraPosition - intersection_point).norm()
        nudged_point = intersection_point + normal * .0001

        # Shadow: check if the point is shadowed or not.
        light_distances = [s.intersect(nudged_point, to_light) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        see_light = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = Color(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lambert_value = np.maximum(normal.dot(to_light), 0)
        color += self.get_diffuse_color(intersection_point) * lambert_value * see_light

        # Reflection
        if bounce < 2:
            reflected_direction = (ray_direction - normal * 2 * ray_direction.dot(normal)).norm()
            color += raytrace(nudged_point, reflected_direction, scene, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = normal.dot((to_light + to_observer).norm())
        color += Color(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * see_light

        return color

class CheckeredSphere(Sphere):
    def diffuse_color(self, intersection_point):
        checker = ((intersection_point.x * 2).astype(int) % 2) == ((intersection_point.z * 2).astype(int) % 2)
        return self.diffuse_color * checker

# Scene setup
scene_objects = [
    Sphere(Vector3D(.75, .1, 1.), .6, Color(0, 0, 1)),
    Sphere(Vector3D(-.75, .1, 2.25), .6, Color(.3, .123, .321)),
    Sphere(Vector3D(-2.75, .1, 3.5), .6, Color(1, .5, .25)),
    CheckeredSphere(Vector3D(0, -99999.5, 0), 99999, Color(.75, .75, .75), 0.25),
]

# Screen coordinates: x0, y0, x1, y1.
aspect_ratio = float(width) / height
screen_coordinates = (-1., 1. / aspect_ratio + .25, 1., -1. / aspect_ratio + .25)
x_coordinates = np.tile(np.linspace(screen_coordinates[0], screen_coordinates[2], width), height)
y_coordinates = np.repeat(np.linspace(screen_coordinates[1], screen_coordinates[3], height), width)

t0 = time.time()
pixel_coordinates = Vector3D(x_coordinates, y_coordinates, 0)
resulting_color = raytrace(CameraPosition, (pixel_coordinates - CameraPosition).norm(), scene_objects)
print("Execution time:", time.time() - t0)

image_array = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((height, width))).astype(np.uint8), "L") for c in resulting_color.components()]
Image.merge("RGB", image_array).show()

# Image.merge("RGB", image_array).save("raytraced_image.png")
