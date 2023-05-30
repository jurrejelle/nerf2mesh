import argparse
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import json
import torch.nn as nn
import colorsys

# 6 -> 32 -> 3
def evaluateNetwork(f0 : np.ndarray, viewdir : np.ndarray, weightsZero, weightsOne):
    v = np.asarray([viewdir[0], viewdir[1], viewdir[2], f0[0], f0[1], f0[2]]).reshape((1,6))
    stage1 = v @ weightsZero
    # Relu
    stage1_relu = np.maximum(0, stage1)

    stage2 = stage1_relu @ weightsOne
    # Sigmoid
    return 1.0 / (1.0 + np.exp(-stage2)); 
    

def convert_to_tex(network_weights):
    width = len(network_weights)
    height = len(network_weights[0])
    
    weightsData = []
    for co in range(0, height):
        for ci in range(0, width):
            weight = network_weights[ci][co]
            weightsData.append(weight)
    
    width_pad = width + (4 - width % 4)
    weightsData_pad = []
    for j in range(0, width_pad, 4):
        for i in range(0, height):
            for c in range(0, 4):
                if (c + j >= width):
                    weightsData_pad.append(0.0)
                else:
                    weightsData_pad.append(weightsData[j + i * width + c])
    return np.asarray(weightsData_pad).reshape(-1,4)

# In here, weightsZero and weightsOne are formatted to change from (32,6) and (6,3) to (64,4) and (6,4) (zero-padded)
def evaluateNetwork_unit(f0 : np.ndarray, viewdir : np.ndarray, weightsZero, weightsOne):
    v = np.asarray([viewdir[0], viewdir[1], viewdir[2], f0[0]]).reshape((4,1))
    results_one = np.zeros((8,4))
    for i in range(0, 32, 4):
        w = np.append(weightsZero[i,:].reshape(4,1),
                      weightsZero[i+1,:].reshape(4,1), axis=1)
        w = np.append(w,
                      weightsZero[i+2,:].reshape(4,1), axis=1)
        w = np.append(w,
                      weightsZero[i+3,:].reshape(4,1), axis=1)
        w = w.reshape(4,4).T
        temp = w @ v
        results_one[i // 4] = results_one[i // 4] + temp.T

    v = np.asarray([f0[1], f0[2], 0, 0]).reshape((4,1))
    for i in range(0, 32, 4):
        w = np.append(weightsZero[32+i,:].reshape(4,1),
                      weightsZero[32+i+1,:].reshape(4,1), axis=1)
        w = np.append(w,
                      weightsZero[32+i+2,:].reshape(4,1), axis=1)
        w = np.append(w,
                      weightsZero[32+i+3,:].reshape(4,1), axis=1)
        w = w.reshape(4,4).T
        temp = w @ v
        results_one[i // 4] = results_one[i // 4] + temp.T
    
    result = np.zeros((1,4))

    for i in range(0, 32//4):
        v = np.maximum(results_one[i], 0.0)
        w = np.append(weightsOne[i*3,:].reshape(4,1),
                      weightsOne[i*3+1,:].reshape(4,1), axis=1)
        w = np.append(w,
                      weightsOne[i*3+2,:].reshape(4,1), axis=1)
        w = np.append(w,
                     np.zeros((4,1)), axis=1)
        w = w.reshape(4,4).T
        temp = w @ v
        result = result + temp.T

    result = result[:,:-1]

    return 1.0 / (1.0 + np.exp(-result))

def rgb2hsv(rgb):
    """ convert RGB to HSV color space

    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

            
"""
vec2 octahedral_mapping(vec3 co)
{
    // projection onto octahedron
	co /= dot( vec3(1), abs(co) );

    // out-folding of the downward faces
    if ( co.y < 0.0 ) {
		co.xy = (1.0 - abs(co.zx)) * sign(co.xz);
    }

	// mapping to [0;1]ˆ2 texture space
	return co.xy * 0.5 + 0.5;
}

vec3 octahedral_unmapping(vec2 co)
{
    co = co * 2.0 - 1.0;

    vec2 abs_co = abs(co);
    vec3 v = vec3(co, 1.0 - (abs_co.x + abs_co.y));

    if ( abs_co.x + abs_co.y > 1.0 ) {
        v.xy = (abs(co.yx) - 1.0) * -sign(co.xy);
    }

    return v;
}

"""
def sign(x):
    return -1 if x < 0 else 1

def octahedral_mapping(view_direction):
    p = view_direction
    
    # Project onto octahedron
    p /= np.abs(p[0]) + np.abs(p[1]) + np.abs(p[2])

    # Out-folding of the downward faces
    r = np.asarray([0,0], dtype=np.float32)
    if ( p[1] < 0.0 ): 
        r[0] = sign(p[0]) * (1-np.abs(-p[2]))
        r[1] = sign(-p[2]) * (1-np.abs(p[0]))
    else:
        r[0] = p[0]
        r[1] = -p[2]


    # This output is in the [-1, 1] space
	# Mapping to [0;1]ˆ2 texture space
    return r * 0.5 + 0.5

def octahedral_unmapping(uv):
    uv = uv * 2 - 1
    v = np.asarray([
        uv[0],
        1 - np.abs(uv[0]) - np.abs(uv[1]),
        -uv[1]
    ], dtype=np.float64)
    if np.abs(uv[0]) + np.abs(uv[1]) > 1:
        v[0] = (np.abs(-uv[1]) - 1) * -sign(uv[0])
        v[2] = (np.abs(uv[0]) - 1) * -sign(-uv[1])

    return v

def check(resolution=512):
    for x in range(resolution):
        for y in range(resolution):

            # Map to [0,1]**2 space
            original_uv = np.asarray([x,y], dtype=np.float32)

            modified_uv = original_uv / resolution

            # Get normalized view direction
            view_direction = norm(octahedral_unmapping(modified_uv))
            
            # Translate back into uvs
            uv = octahedral_mapping(view_direction) * resolution
            if not (original_uv - uv < 1e-7).all():
                print(f"original uv: {original_uv}")
                print(f"view direction: {view_direction}")
                print(f"re-constructed uvs: {uv}")
                print()
                exit()

            


def create_octahedron_mapping(resolution, f0, weightsZero, weightsOne):
    output_array = np.zeros((resolution, resolution, 3))
    for y in range(resolution):
        for x in range(resolution):
            # Compute the direction vector from the pixel coordinates
            original_uv = np.asarray([x,y], dtype=np.float64)
            modified_uv = original_uv / resolution

            # Get normalized view direction
            view_direction = norm(octahedral_unmapping(modified_uv))

            # Pass the direction vector to the evaluate_direction function
            pixel_color = evaluateNetwork(f0, norm(view_direction), weightsZero, weightsOne)

            # Set the output pixel color
            output_array[y, x, :] = pixel_color

    return output_array


def norm(x): 
    return x / np.linalg.norm(x)

# From https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def generateEvenlySpacedPoints(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.column_stack((x,y,z))


identifierArray = generateEvenlySpacedPoints(50)
def generateIdentifier(input_color, weightsZero, weightsOne):
    evalAtDirection = lambda dir: evaluateNetwork(input_color, dir, weightsZero, weightsOne)
    #return rgb2hsv(np.array([evalAtDirection(identDirection) for identDirection in identifierArray]))
    return np.array([evalAtDirection(identDirection) for identDirection in identifierArray])

# Takes in an array, and returns:
#    The unique values
#    The weight of that unique value
#    The indices in the unique array that the original value can be found
def uniques(inputArray):
    size = inputArray.shape[0]
    uniques = np.zeros((size, 3))
    weights = np.zeros(size)
    indices = np.zeros(size, dtype=np.uint32)
    last_color = 0
    color_dict = {}

    for index, color in enumerate(inputArray):
        if tuple(color) in color_dict:
            current_index = color_dict[tuple(color)]
        else:
            current_index = last_color
            uniques[current_index] = color
            color_dict[tuple(color)] = current_index
            last_color += 1

        indices[index] = current_index
        weights[current_index] += 1

    # Cut off the arrays at last_color index
    uniques = uniques[:last_color]
    weights = weights[:last_color]

    return uniques, weights, indices



def precompute(path, clusters=64, mappingResolution=512):
    check()
    stage1_path = os.path.join(path, 'mesh_stage1')

    new_workspace = os.path.join(path, 'mesh_stage2')
    os.makedirs(new_workspace, exist_ok=True) 
    
    # Initialize all the MLP stuff
    mlp_json = json.load(open(os.path.join(stage1_path, "mlp.json"), "r"))
    weightsZero = np.asarray(mlp_json["net.0.weight"])
    weightsOne = np.asarray(mlp_json["net.1.weight"])


    img = cv2.imread(os.path.join(stage1_path, 'feat1_0.png'), cv2.COLOR_BGR2RGB)
    specularArrayOrig = np.asarray(img[:,:])
    resolution = specularArrayOrig.shape[0]
    
    # Assert that our original shape is square and r,g,b
    assert specularArrayOrig.shape == (resolution, resolution, 3)
    
    specularArray = specularArrayOrig.reshape(resolution*resolution, -1)


    # Assert that reshaping back to original size gives original array
    assert np.array_equal(
        specularArray.reshape(resolution, resolution, -1),
        specularArrayOrig)
    
    # Convert the color space from [0,255] to [0,1]
    specularArray = specularArray / 255.0

    # Prepare the dataset since a lot of the colors will be the same
    unique_inputs, weights, indices = uniques(specularArray)

    # Garbage collection to free some memory
    specularArray = None
    
    # Create unique identifiers for each unique datapoint in our input array
    print("Start of generating identifiers")
    unique_outputs = np.array([generateIdentifier(unique_input, weightsZero, weightsOne) for unique_input in unique_inputs])


    # Transform the dataset so it can be used for k-means clustering, namely one big array per pixel
    unique_outputs = unique_outputs.reshape(unique_outputs.shape[0], -1)

    print(f"Done generating identifiers, output shape: {unique_outputs.shape}")

    # Perform Weighted k-means clustering 
    kmeans = KMeans(n_clusters=clusters, random_state=0, verbose=3).fit(unique_outputs, sample_weight=weights)
    weights = None

    # Assign clusters to each input value using the unique_inputs/unique_outputs pair we defined above
    input_clusters = unique_outputs[indices]

    # Get output labels 
    labels = kmeans.predict(input_clusters).reshape(resolution,resolution)
    print("Done assigning labels")

    # Store output labels in clusters of rgb brightness (i / clusters * 255)
    padded_labels = np.zeros((resolution, resolution, 3), dtype=np.float32)
    padded_labels[:,:,0] = labels
    padded_labels[:,:,1] = labels
    padded_labels[:,:,2] = labels
    
    # Change from [0,1] space to [0,255] space
    padded_labels = padded_labels.astype(np.uint8)
    labels_mat = cv2.cvtColor(padded_labels[..., :3], cv2.COLOR_RGB2BGR)
    print("Saving labels_map.png")
    cv2.imwrite(os.path.join(new_workspace, f'labels_map.png'), labels_mat,  [cv2.IMWRITE_PNG_COMPRESSION, 0])

    padded_labels = None
    labels_mat = None

    # Find the closest output to each cluster center
    closest_outputs = []
    cluster_centers = kmeans.cluster_centers_
    kmeans = None

    for center in cluster_centers:
        distances = np.linalg.norm(unique_outputs - center, axis=1)
        closest_output_index = np.argmin(distances)
        closest_output = unique_outputs[closest_output_index]
        closest_outputs.append(closest_output)

    # Find the corresponding input for each closest output
    closest_inputs = unique_inputs[np.isin(unique_outputs, closest_outputs).all(axis=1)]

    print("Done calculating inputs for cluster centers")
    # TODO: FIND NEW METRIC FOR ACCURACY OF THE CLUSTERING
    np.save(os.path.join(new_workspace, f'clusters.json'), closest_inputs)

    unique_outputs = None
    unique_inputs = None
    input_clusters = None

    print("Generating octahedral mappings for clusters:")
    num_clusters = len(closest_inputs)
    mapping_width = int(np.sqrt(num_clusters))
    mappings_final = np.zeros((mapping_width * mappingResolution, mapping_width * mappingResolution, 3), dtype=np.uint8)  # Initialize mappings_final array

    for i, center in enumerate(closest_inputs):
        mapping = create_octahedron_mapping(mappingResolution, center, weightsZero, weightsOne)
        # Convert from [0,1] space to [0,255] space
        mapping *= 255
        mapping = mapping.astype(np.uint8)
        row = i // mapping_width
        col = i % mapping_width
        mappings_final[row * mappingResolution:(row + 1) * mappingResolution, col * mappingResolution:(col + 1) * mappingResolution] = mapping
        print(f"Cluster {i}/{num_clusters} done")
    


    octahedron_mat = cv2.cvtColor(mappings_final[..., :3], cv2.COLOR_RGB2BGR)
    print("Saving octahedron_maps.png")
    cv2.imwrite(os.path.join(new_workspace, f'octahedron_maps.png'), octahedron_mat,  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def check_images(path, clusters=64, mappingResolution=512):
    new_workspace = os.path.join(path, 'mesh_stage2')
    img = cv2.imread(os.path.join(new_workspace, 'labels_map.png'), cv2.COLOR_BGR2RGB)
    labels = np.asarray(img[:,:])
    print(labels[300:310,300:310,:])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--clusters', type=int, default=64)
    parser.add_argument('--mapping_resolution', type=int, default=512)

     
    opt = parser.parse_args()
    precompute(opt.workspace, opt.clusters, opt.mapping_resolution)
    #check_images(opt.workspace, opt.clusters, opt.mapping_resolution)