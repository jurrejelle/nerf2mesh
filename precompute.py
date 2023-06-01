import argparse
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import json
import torch.nn as nn
import colorsys
from PIL import Image
import matplotlib.pyplot as plt

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
def sign(a):
    if(a<=0):
        return -1
    return 1
def octahedral_mapping(v):

    # Project onto octahedron
    v /= np.abs(v[0]) + np.abs(v[1]) + np.abs(v[2])

    # Out-folding of the downward faces
    uv = np.asarray([0,0], dtype=np.float32)
    if ( v[1] < -0.000001 ):
        if sign(v[0]) * sign(v[2]) == 1:
            uv[1] = (1 - np.abs(v[0])) * sign(v[0])
            uv[0] = (1 - np.abs(v[2])) * sign(v[2])
        if sign(v[0]) * sign(v[2]) == -1:
            uv[1] = (1 - np.abs(v[0])) * -sign(v[0])
            uv[0] = (1 - np.abs(v[2])) * -sign(v[2])
    else:
        uv[0] = v[0]
        uv[1] = v[2]

    # This output is in the [-1, 1] space
    # Mapping to [0;1]ˆ2 texture space
    return uv * 0.5 + 0.5

def octahedral_unmapping(uv):
    uv = uv * 2 - 1
    v = np.asarray([
        uv[0],
        1 - np.abs(uv[0]) - np.abs(uv[1]),
        uv[1]
    ], dtype=np.float64)
    if np.abs(uv[0]) + np.abs(uv[1]) > 1:
        if sign(uv[0]) * sign(uv[1]) == 1:
            v[2] = (1 - np.abs(uv[0])) * sign(uv[0])
            v[0] = (1 - np.abs(uv[1])) * sign(uv[1])
        if sign(uv[0]) * sign(uv[1]) == -1:
            v[2] = (1 - np.abs(uv[0])) * -sign(uv[0])
            v[0] = (1 - np.abs(uv[1])) * -sign(uv[1])
    return v


def check(resolution=512):
    for x in range(1,resolution):
        for y in range(1,resolution):

            # Map to [0,1]**2 space
            original_uv = np.asarray([x,y], dtype=np.float32)

            modified_uv = original_uv / resolution

            # Get normalized view direction
            view_direction = octahedral_unmapping(modified_uv)
            
            # Translate back into uvs
            uv = octahedral_mapping(view_direction)
            if (not (modified_uv - uv < 1e-7).all()):
                print(f"original uv: {modified_uv}")
                print(f"view direction: {view_direction}")
                print(f"re-constructed uvs: {uv}")
                print(modified_uv - uv)
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
            view_direction = octahedral_unmapping(modified_uv)
            
            # Pass the direction vector to the evaluate_direction function
            pixel_color = evaluateNetwork(f0, norm(view_direction), weightsZero, weightsOne)
            # pixel_color = view_direction * 0.5 + 0.5
            # Set the output pixel color
            output_array[x, y, :] = pixel_color

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
    color = np.array([evalAtDirection(identDirection) for identDirection in identifierArray])
    hsv = [colorsys.rgb_to_hsv(c[0][0], c[0][1], c[0][2]) for c in color]
    vals = [(c[2] * 9 + c[0] * 3 + c[1]) / 13 for c in hsv]
    # return rgb2hsv(np.array([evalAtDirection(identDirection) for identDirection in identifierArray]))
    return vals

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


def get_labels(unique_outputs, cluster_outputs):
    labels = np.zeros(len(unique_outputs))
    for i in range(len(unique_outputs)):
        minmse = 10000000000000000
        cluster = -1
        for j, clusterdata in enumerate(cluster_outputs):
            error = np.linalg.norm(unique_outputs[i] - clusterdata)
            mse = error / len(clusterdata)
            if(mse < minmse):
                minmse = mse
                cluster = j
        labels[i] = cluster
    return labels

def precompute(path, clusters=64, mappingResolution=512):
    check()
    stage1_path = os.path.join(path, 'mesh_stage1')

    new_workspace = os.path.join(path, 'mesh_stage2')
    os.makedirs(new_workspace, exist_ok=True) 
    
    # Initialize all the MLP stuff
    mlp_json = json.load(open(os.path.join(stage1_path, "mlp.json"), "r"))
    weightsZero = np.asarray(mlp_json["net.0.weight"])
    weightsOne = np.asarray(mlp_json["net.1.weight"])


    file = Image.open(os.path.join(stage1_path, 'feat1_0.png')) #SAYS 2, #TODO FIX
    specularArrayOrig = imageformat_to_data(np.array(file))
    resolution = file.size[0]
    
    # Assert that our original shape is square and r,g,b
    assert specularArrayOrig.shape == (resolution, resolution, 3)

    """
    uvs = (0.1, 0.15)

    viewdir = norm(np.array([0, 0.8, 0.2]))
    (uvx, uvy) = uvs
    texture_sample = specularArrayOrig[int(uvx*resolution)][int(uvy*resolution)]/255
    target = evaluateNetwork(texture_sample, viewdir, weightsZero, weightsOne)
    """

    specularArray = specularArrayOrig.reshape(resolution*resolution, -1)

    testData = generateIdentifier(specularArray[0], weightsZero, weightsOne)

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
    closest_inputs = [unique_inputs[np.where(unique_outputs == x)[0][0]] for x in closest_outputs]

    print("Done calculating inputs for cluster centers")
    # TODO: FIND NEW METRIC FOR ACCURACY OF THE CLUSTERING
    np.save(os.path.join(new_workspace, f'clusters.json'), closest_inputs)

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
        print(f"Cluster {i+1}/{num_clusters} done")

    #     if i == cluster:
    #         test_data = [mappingResolution * octahedral_mapping(dir) for dir in identifierArray]
    #         uv = [[int(x) for x in y] for y in test_data]
    #         colors = [mapping[uvi[0]][uvi[1]] for uvi in uv]
    octahedron_map = Image.fromarray(data_to_imageformat(mappings_final[..., :3]))
    print("Saving octahedron_map.png")
    octahedron_map.save(os.path.join(new_workspace, f'octahedron_maps.png'))
    octahedron_map.close()

    # Assign clusters to each input value using the unique_inputs/unique_outputs pair we defined above
    # input_clusters = unique_outputs[indices]

    # Get output labels 
    
    labels = get_labels(unique_outputs, closest_outputs)[indices].reshape(resolution, resolution)
    # labels = kmeans.predict(input_clusters).reshape(resolution,resolution)
    print("Done assigning labels")

    # Store output labels in clusters of rgb brightness (i)
    padded_labels = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    padded_labels[:,:,0] = labels
    padded_labels[:,:,1] = labels
    padded_labels[:,:,2] = labels
    
    # Change from [0,1] space to [0,255] space
    labels_image = Image.fromarray(data_to_imageformat(padded_labels[..., :3]))
    print("Saving labels_map.png")
    labels_image.save(os.path.join(new_workspace, f'labels_map.png'))
    labels_image.close()

    """
    for i in range(num_clusters):
        label = i
        lx = label % mapping_width
        ly = label // mapping_width
        uv2 = octahedral_mapping(viewdir)
        pixelx = int((lx + uv2[0]) * mappingResolution)
        pixely = int((ly + uv2[1]) * mappingResolution)
        value = mappings_final[pixelx][pixely]
        print(texture_sample)
        print(closest_inputs[i])
        print(f"label: {i}")
        print(f"target: {target*255}")
        print(f"value: {value}")
    """


def data_to_imageformat(arr):
    newArr  = np.swapaxes(arr, 0, 1) # [x][y][c]
    newArr = newArr[::-1,:,:3] # [y][x][c]
    return newArr


def imageformat_to_data(arr):
    newArr = arr[::-1,:,:3] # [y][x][c]
    newArr  = np.swapaxes(newArr, 0, 1) # [x][y][c]
    return newArr

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
    