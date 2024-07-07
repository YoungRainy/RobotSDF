
#%%
import numpy as np
import torch
import os
import json
import deep_sdf.workspace as ws
import deep_sdf
import random

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
from math import cos
from math import sin
import matplotlib.pyplot as plt
import timeit
import pyvista as pv

import deep_sdf.utils
#%%

def read_sdf_samples_into_ram(filename,sampleNum):
    npz = np.load(filename)
    data_set = np.vstack((npz['pos'], npz['neg']))
    random.shuffle(data_set)
    rows_random = np.random.choice(data_set.shape[0],size=sampleNum,replace=False)
    data_set_=data_set[rows_random,:]
    # data_set_=torch.from_numpy(data_set_)
    return data_set_


def points_trans(points,rotation_matrix):
     tran_points = torch.zeros(len(points),3)
     ones = torch.ones(1,len(points))
     tran_points[:,0] = rotation_matrix[0,0] * points[:,0] + rotation_matrix[0,1] * points[:,1] + rotation_matrix[0,2] * points[:,2] +rotation_matrix[0,3] * ones 
     tran_points[:,1] = rotation_matrix[1,0] * points[:,0] + rotation_matrix[1,1] * points[:,1] + rotation_matrix[1,2] * points[:,2] +rotation_matrix[1,3] * ones
     tran_points[:,2] = rotation_matrix[2,0] * points[:,0] + rotation_matrix[2,1] * points[:,1] + rotation_matrix[2,2] * points[:,2] +rotation_matrix[2,3] * ones
     return tran_points  

def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)
    
    sdf = decoder(inputs)

    return sdf
def running_time(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)
    start_time=time.time()
    sdf = decoder(inputs)
    end_time=time.time()
    total_time = end_time - start_time
    print("collision dection time:",total_time," sec")
    return total_time
    
def matrix_multiply(matrix1, matrix2):
    # 检查矩阵维度是否匹配
    if len(matrix1[0]) != len(matrix2):
        print("矩阵维度不匹配，无法进行乘法。")
        return None

    result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result
def link_trans_in_base(theta):
    Z1 = 0.031;X2 = -0.068808;Z2 =  0.1615;Z3 = 0.266;Z4 = -0.10523;X4 = 0.068808;Z5 = 0.21877;Z6 = -0.1065;Z7 = -0.091 #elfin3
    # z1 = 0.03; X2 = -0.078069;Z2 = 0.19 ; Z3 = 0.38;Z4 = 0.078069;X4 = -0.11342; Z5 = 0.30658; Z6 = -0.1065; Z7 = -0.0735 #elfin5
# //                                    ||                                    第一行                                        ||     ||                                  第二行                                                                ||                                          第三行                                             ||        || 第四行|| 
    base_to_link1 = np.array([[cos(theta[1]),0,sin(theta[1]),0],[sin(theta[1]),0,-cos(theta[1]),0],[0,1,0,Z1],[0, 0, 0, 1]])
    link1_to_link2 = np.array([[1,0,0,X2], [0,cos(theta[2]),sin(theta[2]),Z2],[0,-sin(theta[2]),cos(theta[2]),0],[0, 0,  0, 1]])
    link2_to_link3 = np.array([[1,0,0,0],[0,-cos(theta[3]),sin(theta[3]),Z3],[0,-sin(theta[3]),-cos(theta[3]),0],[0, 0, 0, 1]])
    link3_to_link4 = np.array([[cos(theta[4]),0,sin(theta[4]),X4],[0,-1,0,Z4],[sin(theta[4]),0,-cos(theta[4]),0],[0, 0, 0, 1]])
    link4_to_link5 = np.array([[1,0,0,0],[0,-cos(theta[5]),sin(theta[5]),Z5],[0,-sin(theta[5]),-cos(theta[5]),0],[0, 0, 0, 1]])
    link5_to_link6 = np.array([[-cos(theta[6]),0,sin(theta[6]),0],[0,1,0,Z6],[-sin(theta[6]),0,-cos(theta[6]),0],[0, 0, 0, 1]])
    link6_to_end   = np.array([[0,-1,0,0],[0,0,-1,Z7],[1,0,0,0],[0, 0, 0, 1]])

    # base_to_link2 = base2link1 * link1_to_link2
    base_to_link2 = matrix_multiply(base_to_link1,link1_to_link2)
    base_to_link3 = matrix_multiply(base_to_link2,link2_to_link3)
    base_to_link4 = matrix_multiply(base_to_link3,link3_to_link4)
    base_to_link5 = matrix_multiply(base_to_link4,link4_to_link5)
    base_to_link6 = matrix_multiply(base_to_link5,link5_to_link6)

    link1_to_base = np.linalg.inv(base_to_link1)
    link2_to_base = np.linalg.inv(base_to_link2)
    link3_to_base = np.linalg.inv(base_to_link3)
    link4_to_base = np.linalg.inv(base_to_link4)
    link5_to_base = np.linalg.inv(base_to_link5)
    link6_to_base = np.linalg.inv(base_to_link6)

    return np.array([link1_to_base,link2_to_base,link3_to_base,link4_to_base,link5_to_base,link6_to_base])

def create_mesh(
    decoder,filename,model_parameter,code,theta,N=256, max_batch=32 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    
    rotation_matrix = torch.tensor(link_trans_in_base(theta))

    num_samples = N ** 3

    samples.requires_grad = False

    sdf_link1_6 = torch.zeros(len(samples),len(code))
    for i in range(len(code)):
        head = 0
        decoder.load_state_dict(model_parameter[i])
        latent_vec = code[i]
        tran_samples = points_trans(samples,rotation_matrix[i])
        while head < num_samples:
            sample_subset = tran_samples[head : min(head + max_batch, num_samples), 0:3].cuda()

            samples[head : min(head + max_batch, num_samples), 3] = (
                deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
                .squeeze(1)
                .detach()
                .cpu()
            )
            head += max_batch
        sdf_link1_6[:,i] = samples[:,3]
    sdf_values = torch.min(sdf_link1_6,dim=1).values
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
  
def load_model_parameter_code(state_path):
    with open(state_path, 'r') as file:
        state_path = json.load(file)
    root_path = list(state_path.keys())[0]
    state_root_path = list(state_path[root_path].keys())[0]
    state_name = state_path[root_path][state_root_path]
    model_parameter = []
    code = []
    model_name = [name[:-4] for name in state_name]
    for i in range(len(state_name)):
        parameter_path=os.path.join(root_path, 'parameter', state_name[i])
        code_path=os.path.join(root_path,'code',state_name[i])
        saved_model_state = torch.load(parameter_path)['model_state_dict']
        model_parameter.append(saved_model_state)
        code.append(torch.load(code_path)[0])
    return model_name,model_parameter,code
def preSdfElfin3(decoder,model_parameter,code,Queries,theta):
    decoder.eval()
    rotation_matrix = link_trans_in_base(theta)
    sdf_link1_6 = torch.zeros(len(queries),len(code))
    for i in range(len(code)):
        decoder.load_state_dict(model_parameter[i])
        latent_vec = code[i]
        tran_queries = points_trans(Queries,rotation_matrix[i]).cuda()
        sdf_link1_6[:,i] = decode_sdf(decoder, latent_vec, tran_queries).squeeze(1).detach().cpu()
    # print(sdf_link1_6)
    pre_sdf = torch.min(sdf_link1_6,dim=1).values
    return pre_sdf
def load_txt_dataset(file_path, delimiter=','):
    """
    Load a text file as a dataset.

    Args:
        file_path (str): The path to the text file containing the dataset.
        delimiter (str, optional): The delimiter used to separate values in each line.

    Returns:
        np.ndarray: A NumPy array containing the dataset.
    """
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual values using the specified delimiter
            values = line.strip().split(delimiter)
            # Convert values to appropriate data types if needed
            # For example, you can use float(values[i]) to convert a value to a float
            # Append the values as a sample to the dataset
            sample = [float(value) for value in values]
            dataset.append(sample)
    return np.array(dataset,dtype=np.float32)
    


#%%
if __name__ == "__main__":
    #%%
    #定义网络模型
    specs_filename = 'networkSpecs.json'
    # specs_filename = os.path.join(experiment_directory, ".json")
    if not os.path.isfile(specs_filename):
            raise Exception(
                'The experiment directory does not include specifications file "specs.json"'
            )
    specs = json.load(open(specs_filename))
    arch = __import__("networks." + 'deep_sdf_decoder', fromlist=["Decoder"])
    latent_size = 256
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)

    #%%
    #加载网络参数和code
    state_path = './model_state/path.json'
    model_name, model_parameter, code = load_model_parameter_code(state_path)
    #%%
    #重构机械臂模型
    mesh_filename = './reconstruct_mesh' #重构的机械臂模型名称
    config = [-1,0.2,-0.2,0.7,3,-0.8,-1.7] #7个元素，第一个元素无用，后面表示六个关节角度
    theta = np.around(config,6)
    filename = 're_config'
    for i in range(6):
        filename = filename+'_'+str(theta[i+1])
    mesh_filename = os.path.join(mesh_filename,filename)
    with torch.no_grad():
        create_mesh(
            decoder, mesh_filename,model_parameter,code,theta,N=256, max_batch=int(2 ** 18)
        )
    



