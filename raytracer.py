import json
from myshapes import Sphere, Triangle, Plane
import numpy as np
import matplotlib.pyplot as plt

scene_fn = "scene_5.json"
res = 256

#### Scene Loader

def loadScene(scene_fn):

	with open(scene_fn) as f:
		data = json.load(f)

	spheres = []

	for sphere in data["Spheres"]:
		spheres.append(
			Sphere(sphere["Center"], sphere["Radius"], 
		 	sphere["Mdiff"], sphere["Mspec"], sphere["Mgls"], sphere["Refl"],
		 	sphere["Kd"], sphere["Ks"], sphere["Ka"]))
		
	triangles = []

	for triangle in data["Triangles"]:
		triangles.append(
			Triangle(triangle["A"], triangle["B"], triangle["C"],
			triangle["Mdiff"], triangle["Mspec"], triangle["Mgls"], triangle["Refl"],
			triangle["Kd"], triangle["Ks"], triangle["Ka"]))
	
	planes = []

	for plane in data["Planes"]:
		planes.append(
			Plane(plane["Normal"], plane["Distance"],
			plane["Mdiff"], plane["Mspec"], plane["Mgls"], plane["Refl"],
			plane["Kd"], plane["Ks"], plane["Ka"]))
	
	objects = spheres + triangles + planes

	camera = {
		"LookAt": np.array(data["Camera"]["LookAt"],),
		"LookFrom": np.array(data["Camera"]["LookFrom"]),
		"Up": np.array(data["Camera"]["Up"]),
		"FieldOfView": data["Camera"]["FieldOfView"]
	}

	light = {
		"DirectionToLight": np.array(data["Light"]["DirectionToLight"]),
		"LightColor": np.array(data["Light"]["LightColor"]),
		"AmbientLight": np.array(data["Light"]["AmbientLight"]),
		"BackgroundColor": np.array(data["Light"]["BackgroundColor"]),
	}

	return camera, light, objects

### Ray Tracer

camera, light, objects = loadScene(scene_fn)

image = np.zeros((res,res,3), dtype=np.float32)


# YOUR CODE HERE
#Step 2
def cast_rays(ray_org, ray_dir):
	tmin = -1
	closest_obj = None
	for obj in objects:
		t = obj.intersect(ray_org, ray_dir)
		if t > 0 and (t < tmin or tmin == -1):
			tmin = t
			closest_obj = obj
	return tmin, closest_obj

def getColor(r0, rd, max_bounces = 4):
	tmin, obj = cast_rays(r0, rd)
	if tmin > 0 :
		p = r0 + rd * tmin
		if isinstance(obj, Sphere):
			normal = obj.getNormal(p)
		else:
			normal = obj.getNormal()
		normal = normal / np.linalg.norm(normal)

		l = light["DirectionToLight"] / np.linalg.norm(light["DirectionToLight"])
		shadow_ray_dir = l
		shadow_ray_origin = p + normal * 1e-5 
		shadow_tmin, shadow_obj = cast_rays(shadow_ray_origin, shadow_ray_dir)
		in_shadow = shadow_obj is not None and shadow_tmin > 0.01

		cdiff = np.zeros(3)
		cspec = np.zeros(3)
		Crefl = np.zeros(3)
  
		if not in_shadow:
			diffuse = np.dot(normal , l)
			diffuse = max(diffuse, 0)
			cdiff = np.multiply(light["LightColor"], obj.getDiffuse()) * diffuse
			cspec = np.zeros(3)
	
			if diffuse > 0:
				r = 2 * np.dot(l, normal) * normal - l
				r = r / np.linalg.norm(r)
				spec = max(np.dot(-rd, r), 0) ** obj.getGloss()
				cspec = np.multiply(light["LightColor"], obj.getSpecular()) * spec
			if obj.getRefl() > 0:
				Crefl_dir = rd -2 * np.dot(normal, rd) * normal
				Crefl_dir = Crefl_dir / np.linalg.norm(Crefl_dir)
				Crefl_r0 = p + 0.001 * normal

				if max_bounces > 0:
					Crefl = getColor(Crefl_r0, Crefl_dir, max_bounces - 1)
				else: 
					Crefl = light["BackgroundColor"]
    			
		
		camb = np.multiply(light["AmbientLight"], obj.getDiffuse())
		color = obj.getKd() * cdiff + obj.getKs() * cspec + obj.getKa() * camb + obj.getRefl() * Crefl
		color = np.clip(color, 0, 1)
	else: 
		color = light["BackgroundColor"]
	return color 

#Step 1
#Gram-schmidt 
e3 = (camera["LookAt"]- camera["LookFrom"]) / np.linalg.norm(camera["LookAt"]- camera["LookFrom"])
e1 = np.cross(e3, camera["Up"]) / np.linalg.norm(np.cross(e3, camera["Up"]) )
e2 = np.cross(e1, e3) / np.linalg.norm(np.cross(e1, e3) )

# Determine the dimensions of the window based on the field of view.
fov = np.deg2rad(camera["FieldOfView"])
d = np.linalg.norm(camera["LookAt"]- camera["LookFrom"])
Umax = d * np.tan(fov / 2)
Vmax = d * np.tan(fov / 2)
Umin = -Umax 
Vmin = -Vmax

# Determine the s coordinate for each pixel based on the resolution (start with 256x256).
# Set the ray origin ro to be camera["LookFrom"] and calculate the ray direction rd to be e1*s + e2*t + e3*d
du = (Umax - Umin) / (res + 1)
dv = (Vmax - Vmin) / (res + 1)

r0 = camera["LookFrom"]


for i in range(int(-res/2), int(res/2)):
	for j in range(int(-res/2), int(res/2)):
		s = camera["LookAt"] + du * (j + 0.5) * e1 + dv * (i + 0.5) * e2
		rd = (s-r0) / np.linalg.norm(s-r0)
		
		color = getColor(r0, rd)
		image[i + int(res/2), j + int(res/2)] = color

image = np.flip(image, axis=0)

### Save and Display Output
plt.imsave("output.png", image)
plt.imshow(image);plt.show()

