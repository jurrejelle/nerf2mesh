import argparse
import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import json
from PIL import Image
import time
from datetime import datetime
import shutil 

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

# In here, weightsZero and weightsOne are formatted to change from (32,6) and (6,3) to (64,4) and (6,4) (zero-padded) using the convert_to_tex method
# evaluateNetwork_unit(f0, viewdir, convert_to_tex(weightsZero), convert_to_tex(weightsOne)) 
# === 
# evaluateNetwork(f0, viewdir, weigthsZero, weightsOne)
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


def export_data(times, clusters, deduped, resolution, mapping_resolution, unique_points, precompute_identifiers, evaluate_identifiers, MAE, MSE, path):
    new_workspace = os.path.join(path, f"results_json")
    os.makedirs(new_workspace, exist_ok=True) 
    # MAE TIME IS NOT INCLUDED IN TOTAL TIME
    output_object = {
        "clusters": clusters,
        "final_clusters": deduped,
        "mapping_resolution": mapping_resolution,
        "input_resolution": resolution,
        "num_precompute_identifiers": precompute_identifiers,
        "num_eval_identifiers": evaluate_identifiers,
        "mean_average_error": MAE,
        "mean_squared_error": MSE,
        "unique_points": unique_points,
        "uniques_time": times["end_uniques"] - times["start_uniques"],
        "identifiers_time": times["end_identifiers"] - times["start_identifiers"],
        "kmeans_time": times["end_kmeans"] - times["start_kmeans"],
        "cluster_finding_time": times["end_cluster_finding"] - times["start_cluster_finding"],
        "octahedral_map_time": times["end_octahedral_map"] - times["start_octahedral_map"],
        "labels_assigning_time": times["end_assigning_labels"] - times["start_assigning_labels"],
        "deduplicating_time": times["end_deduplicating"] - times["start_deduplicating"],
        "total_time": times["end"] - times["start"],
        "mae_time": times["mae_end"] - times["mae_start"]
    }
    with open(new_workspace+"/"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.json"), "w") as f:
        f.write(json.dumps(output_object))

# Custom sign function to prevent division by zero
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


def check_bidirectionality_of_octahedral_map(resolution=512):
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


def generateIdentifier(input_color, weightsZero, weightsOne, identifier_array):
    evalAtDirection = lambda dir: evaluateNetwork(input_color, dir, weightsZero, weightsOne)
    return np.array([evalAtDirection(identDirection) for identDirection in identifier_array])

# Takes in an array, and returns:
#    The unique values
#    The weight of that unique value
#    The indices in the unique array that the original value can be found
#    (e.g. uniques[indices] === inputArray)
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

def precompute(path, clusters, mapping_resolution, output_dir, precompute_identifiers, evaluate_identifiers, eval, copy_old):
    times = {}
    times["start"] = time.time()

    stage1_path = os.path.join(path, 'mesh_stage1')
    new_workspace = os.path.join(path, output_dir)
    os.makedirs(new_workspace, exist_ok=True) 

    if copy_old:
        shutil.copytree(stage1_path, new_workspace, dirs_exist_ok=True)
    else:
        for filename in ["mesh_0.obj", "feat0_0.png"]:
            shutil.copy2(os.path.join(stage1_path, filename), new_workspace)
    # Initialize all the MLP stuff
    mlp_json = json.load(open(os.path.join(stage1_path, "mlp.json"), "r"))
    weightsZero = np.asarray(mlp_json["net.0.weight"])
    weightsOne = np.asarray(mlp_json["net.1.weight"])


    file = Image.open(os.path.join(stage1_path, 'feat1_0.png'))
    specularArrayOrig = imageformat_to_data(np.array(file))
    resolution = file.size[0]
    
    # Assert that our original shape is square and r,g,b
    assert specularArrayOrig.shape == (resolution, resolution, 3)

    specularArray = specularArrayOrig.reshape(resolution*resolution, -1)
    
    # Convert the color space from [0,255] to [0,1]
    specularArray = specularArray / 255.0

    # Prepare the dataset since a lot of the colors will be the same
    print("Finding unique colors to cluster on")
    times["start_uniques"] = time.time()
    unique_inputs, weights, indices = uniques(specularArray)
    print(f"Found {unique_inputs.shape[0]} unique colors")
    times["end_uniques"] = time.time()

    assert sum(weights) == resolution * resolution, "The sum of all weights should equal the original image resolution, {resolution} x {resolution}"

    # Garbage collection to free some memory
    specularArray = None
    
    # Create unique identifiers for each unique datapoint in our input array
    print(f"Generating {precompute_identifiers} identifying points for each color")
    times["start_identifiers"] = time.time()
    identifier_array = generateEvenlySpacedPoints(precompute_identifiers)
    unique_outputs = np.array([generateIdentifier(unique_input, weightsZero, weightsOne, identifier_array) for unique_input in unique_inputs])
    times["end_identifiers"] = time.time()

    # Transform the dataset so it can be used for k-means clustering, namely one big array per pixel
    unique_outputs = unique_outputs.reshape(unique_outputs.shape[0], -1)

    print(f"Done generating identifiers, output shape: {unique_outputs.shape}")

    # Perform Weighted k-means clustering 
    print(f"Performing K-Means clustering to create {clusters} clusters")
    times["start_kmeans"] = time.time()
    kmeans = KMeans(n_clusters=clusters, random_state=0, verbose=3).fit(unique_outputs, sample_weight=weights)
    times["end_kmeans"] = time.time()


    # Find the closest output to each cluster center
    closest_outputs = []
    cluster_centers = kmeans.cluster_centers_
    print("Finding inputs corresponding to the cluster centers")
    times["start_cluster_finding"] = time.time()
    for center in cluster_centers:
        distances = np.linalg.norm(unique_outputs - center, axis=1)
        closest_output_index = np.argmin(distances)
        closest_output = unique_outputs[closest_output_index]
        closest_outputs.append(closest_output)

    # Find the corresponding input for each closest output
    closest_inputs = np.asarray([unique_inputs[np.where(unique_outputs == x)[0][0]] for x in closest_outputs])
    times["end_cluster_finding"] = time.time()


    # Assign clusters to each input value using the unique_inputs/unique_outputs pair we defined above
    input_clusters = unique_outputs[indices]

    # Get output labels 
    print("Assigning output labels")
    times["start_assigning_labels"] = time.time()
    labels = kmeans.predict(input_clusters)
    times["end_assigning_labels"] = time.time()

    # Start de-duplicating of mlp inputs
    print("De-duplicating output clusters")
    times["start_deduplicating"] = time.time()
    
    closest_inputs_uniques, _, closest_inputs_indices = uniques(closest_inputs)
    # Converting from "labels to closest_inputs" to "labels to closest_inputs_uniques"
    labels = closest_inputs_indices[labels]

    times["end_deduplicating"] = time.time()
    print(f"Done de-duplicating, new amount of clusters: {closest_inputs_uniques.shape[0]}")

    np.save(os.path.join(new_workspace, f'clusters'), closest_inputs)
    with open(os.path.join(new_workspace, f'clusters.txt'), "w") as f:
        f.write(str(closest_inputs_uniques.shape[0]))

    print("Generating octahedral mappings for clusters")
    times["start_octahedral_map"] = time.time()
    num_clusters = closest_inputs_uniques.shape[0]
    mapping_width = int(np.ceil(np.sqrt(num_clusters)))
    mappings_final = np.zeros((mapping_width * mapping_resolution, mapping_width * mapping_resolution, 3), dtype=np.uint8)  # Initialize mappings_final array

    for i, center in enumerate(closest_inputs_uniques):
        mapping = create_octahedron_mapping(mapping_resolution, center, weightsZero, weightsOne)
        # Convert from [0,1] space to [0,255] space
        mapping *= 255
        mapping = mapping.astype(np.uint8)
        row = i // mapping_width
        col = i % mapping_width
        mappings_final[row * mapping_resolution:(row + 1) * mapping_resolution, col * mapping_resolution:(col + 1) * mapping_resolution] = mapping
        print(f"Cluster {i+1}/{num_clusters} done")

    times["end_octahedral_map"] = time.time()

    octahedron_map = Image.fromarray(data_to_imageformat(mappings_final[..., :3]))
    print("Saving octahedron_map.png")
    octahedron_map.save(os.path.join(new_workspace, f'octahedron_maps.png'))
    octahedron_map.close()

    # Store output labels of clusters in rgb brightness
    labels = labels.reshape(resolution,resolution)
    padded_labels = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    padded_labels[:,:,0] = (labels % 256)
    padded_labels[:,:,1] = (labels // 256) 
    padded_labels[:,:,2] = (labels // (256 * 256)) 
    
    # Change from [0,1] space to [0,255] space
    labels_image = Image.fromarray(data_to_imageformat(padded_labels[..., :3]))
    print("Saving labels_map.png")
    labels_image.save(os.path.join(new_workspace, f'labels_map.png'))
    labels_image.close()
    times["end"] = time.time()

    total_MAE = 0
    total_MSE = 0
    if eval:
        print("Calculating MAE and MSE")
        times["mae_start"] = time.time()
        # Calculate MAE according to weight, to get MAE accross the entire image
        evaluation_identifier_array = generateEvenlySpacedPoints(evaluate_identifiers)
        for unique_input, weight, unique_output in zip(unique_inputs, weights, unique_outputs):
            # Generate Mean Average Error for the current pixel
            original_label = kmeans.predict(unique_output.reshape(1,-1))[0]
            label = closest_inputs_indices[original_label]
            pixel_MAE = 0
            pixel_MSE = 0
            for evaluation_direction in evaluation_identifier_array:
                # Get actual color, convert it to 0-255 space to be in the same format as the final output
                actual_color = evaluateNetwork(unique_input, evaluation_direction, weightsZero, weightsOne)
                actual_color = actual_color * 255
                actual_color = actual_color.astype(np.int16)

                # Get color by indexing into the final label map array
                uv_x, uv_y = octahedral_mapping(evaluation_direction)

                atlas_x,atlas_y = label // mapping_width, label % mapping_width
                # uv_x is [0,1]
                # we want uv_x to be [0,mapping_resolution-1], so that it doesn't cross over into the next texture

                final_x = int(atlas_x * mapping_resolution + uv_x * (mapping_resolution-1))
                final_y = int(atlas_y * mapping_resolution + uv_y * (mapping_resolution-1))
                
                our_color = mappings_final[final_x, final_y, :].astype(np.int16)

                pixdir_MAE = 0
                pixdir_MSE = 0
                for i in range(3):
                    absval = abs(actual_color[0,i] - our_color[i])
                    pixdir_MAE += absval
                    pixdir_MSE += absval**2

                pixel_MAE += pixdir_MAE / 3
                pixel_MSE += pixdir_MSE / 3

            pixel_MSE /= evaluate_identifiers
            pixel_MAE /= evaluate_identifiers

            total_MAE += pixel_MAE * weight
            total_MSE += pixel_MSE * weight
        total_MAE /= resolution * resolution
        total_MSE /= resolution * resolution
        times["mae_end"] = time.time()
        print(f"Total Mean Average Error accross entire input texture: {total_MAE}")
        print(f"Total Mean Squared Error accross entire input texture: {total_MSE}")
    else:
        times["mae_end"] = 0
        times["mae_start"] = 0
        
    export_data(times, clusters, closest_inputs_uniques.shape[0], resolution, mapping_resolution, len(unique_inputs), precompute_identifiers, evaluate_identifiers, total_MAE, total_MSE, path)
    print("All done :)")

def data_to_imageformat(arr):
    newArr  = np.swapaxes(arr, 0, 1) # [x][y][c]
    newArr = newArr[::-1,:,:3] # [y][x][c]
    return newArr


def imageformat_to_data(arr):
    newArr = arr[::-1,:,:3] # [y][x][c]
    newArr  = np.swapaxes(newArr, 0, 1) # [x][y][c]
    return newArr
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--clusters', type=int, default=64)
    parser.add_argument('--mapping_resolution', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default="mesh_stage2")
    parser.add_argument('--idents', type=int, default=100)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--eval_idents', type=int, default=100)
    parser.add_argument('--copy_old', type=bool, default=False)
    

    
    opt = parser.parse_args()
    precompute(opt.workspace, opt.clusters, opt.mapping_resolution, opt.output_dir, opt.idents, opt.eval_idents, opt.eval, opt.copy_old)
