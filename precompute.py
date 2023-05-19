import argparse
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import json
import torch.nn as nn

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

def precompute(path, clusters=64, mappingResolution=512):
    check()
    stage1_path = os.path.join(path, 'mesh_stage1')
    img = cv2.imread(os.path.join(stage1_path, 'feat1_0.png'), cv2.COLOR_BGR2RGB)
    specularArrayOrig = np.asarray(img[:,:])
    print(specularArrayOrig[0,0])

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
    unique, counts = np.unique(specularArray, axis=0, return_counts=True)

    # Perform Weighted k-means clustering 
    kmeans = KMeans(n_clusters=clusters, random_state=0, verbose=3).fit(unique, sample_weight=counts)

    # Get output labels
    labels = kmeans.predict(specularArray).reshape(resolution,resolution)

    # Store output labels in clusters of rgb brightness (i / clusters * 255)
    padded_labels = np.zeros((resolution, resolution, 3), dtype=np.float32)
    padded_labels[:,:,0] = labels
    padded_labels[:,:,1] = labels
    padded_labels[:,:,2] = labels

    # Get output centers
    centers = kmeans.cluster_centers_
    
    # Generate data based on just closest cluster:
    mean_absolute_error = np.asarray([0,0,0], dtype=np.float64)
    for i in range(resolution):
        for j in range(resolution):
            currentLabel = labels[i,j]
            currentCluster = centers[currentLabel] * 255.0
            originalColor = specularArrayOrig[i,j,:]
            colordiff = abs(currentCluster - originalColor)
            mean_absolute_error += colordiff
    mean_absolute_error = mean_absolute_error / (resolution**2)

    print(f"Mean Absolute Error based on color from first cluster: {mean_absolute_error}")


    # Initialize all the MLP stuff
    mlp_json = json.load(open(os.path.join(stage1_path, "mlp.json"), "r"))
    weightsZero = np.asarray(mlp_json["net.0.weight"])
    weightsOne = np.asarray(mlp_json["net.1.weight"])

    mappings = []
    for i, center in enumerate(centers):
        mappings.append(create_octahedron_mapping(mappingResolution, center, weightsZero, weightsOne))
        print(f"Cluster {i}/{len(centers)} done")

    
    mapping_width = int(np.sqrt(clusters))
    mapping_rows = []
    print(f"Concatinating all {clusters} clusters into one atlas")
    for i in range(mapping_width):
        mapping_rows.append(np.concatenate(mappings[i*mapping_width : (i+1) * mapping_width], axis=1))
    mappings_final = np.concatenate(mapping_rows)

    # Change from [0,1] space to [0,255] space
    mappings_final *= 255

    mappings_final = mappings_final.astype(np.uint8)
    padded_labels = padded_labels.astype(np.uint8)


    octahedron_mat = cv2.cvtColor(mappings_final[..., :3], cv2.COLOR_RGB2BGR)
    labels_mat = cv2.cvtColor(padded_labels[..., :3], cv2.COLOR_RGB2BGR)

    new_workspace = os.path.join(path, 'mesh_stage2')
    os.makedirs(new_workspace, exist_ok=True) 

    print("Saving octahedron_maps.png")
    cv2.imwrite(os.path.join(new_workspace, f'octahedron_maps.png'), octahedron_mat,  [cv2.IMWRITE_PNG_COMPRESSION, 0]) 
    #cv2.imwrite(os.path.join(new_workspace, f'octahedron_maps.jpg'), octahedron_mat) 
    print("Saving labels_map.png")
    cv2.imwrite(os.path.join(new_workspace, f'labels_map.png'), labels_mat,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #cv2.imwrite(os.path.join(new_workspace, f'labels_map.jpg'), labels_mat)

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