from torch.utils import data
import config
import torch
import numpy as np

class VolumeData(data.Dataset):
    def __init__(self, df, dirpath, transform,
                    acc_pdb,test = False):
        
        self.df = df
        self.directory_precomputed = config.PRECOMPUTED_PATH
        self.directory_pdb = config.PDB_PATH
        self.flips = config.FLIPS
        self.acc_labels = df.values.tolist()
        self.acc_pdb = acc_pdb
        self.max_radius = config.MAX_RADIUS
        self.noise_treatment = config.NOISE_TREATMENT
        self.n_channels = max(1, config.INPUT_CHANNELS])
        self.p = config.P
        self.scaling_weights = config.SCALING_WEIGHTS
        self.shuffle = config.SHUFFLE
        self.v_size = config.V_SIZE
        self.weights = config.WEIGHTS
        self.on_epoch_end()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
#         acc = list(self.acc_pdb)
#         pdb_id = self.acc_pdb[acc[idx]]
        X, y = self.__data_augmentation(idx)
        
        label_tensor = torch.zeros((1, output_dim))
        z = torch.from_numpy(y)
        for j,ele in enumerate(z):
            label_tensor[0, j] = ele
        image_label = torch.tensor(label_tensor,dtype= torch.float32)
        
        #   convert X to tensor
        
        return (X, image_label.squeeze())
    
    def __data_augmentation(self, idx):
        'Returns augmented data with batch_size enzymes' # X : (v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.n_channels,
                      self.v_size,
                      self.v_size,
                      self.v_size))
        try:
                   
            y = np.empty(len(self.acc_labels[idx][1]), dtype=int)

            # Computations
            if len(self.acc_labels[idx]) != 2:
                print(len(self.labels[idx]), end = ' ')
            y = self.acc_labels[idx][1]
            pdb_id = self.acc_pdb[self.acc_labels[idx][0]]

            # Load precomputed coordinates
            coords = load_coords(pdb_id, self.p, self.directory_precomputed)
            coords = coords_center_to_zero(coords)
            coords = adjust_size(coords, v_size=self.v_size, max_radius=self.max_radius)

            # Get weights
            local_weights = []
            for weight in self.weights:
                local_weight = load_weights(pdb_id, weight, self.p,
                                            self.scaling_weights, self.directory_precomputed) # Compute extended weights
                local_weights += [local_weight] # Store
                

            # PCA
            coords = PCA(n_components=3).fit_transform(coords)

            # Do flip
            coords_temp = flip_around_axis(coords, axis=self.flips)

            if len(self.weights) == 0:
                # Convert to volume and store
                X[0, :, :, :] = coords_to_volume(coords_temp, self.v_size,
                                                    noise_treatment=self.noise_treatment)

            else:
                # Compute to weights of volume and store
                for k in range(self.n_channels):
                    X[k, :, :, :] = weights_to_volume(coords_temp, local_weights[k],
                                                         self.v_size, noise_treatment=self.noise_treatment)

            return X, np.array(y)
        except:
            print(idx)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.acc_pdb))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
def load_coords(pdb_id, desired_p, source_path):
    'Loads precomputed coordinates'
    return np.load(precomputed_name(pdb_id, source_path, 'coords', desired_p))
    
def coords_center_to_zero(coords):
    'Centering coordinates on [0,0,0]'
    barycenter = get_barycenter(coords)
    return coords - np.full((coords.shape[0], 3), barycenter)

def adjust_size(coords, v_size=32, max_radius=40):
    return np.multiply((v_size/2-1)/max_radius, coords)

def load_weights(pdb_id, weights_name, desired_p, scaling, source_path):
    'Loads precomputed weights'
    return np.load(precomputed_name(pdb_id, source_path, 'weights', desired_p, weights_name, scaling))

def flip_around_axis(coords, axis=(0.2, 0.2, 0.2)):
    'Flips coordinates randomly w.r.t. each axis with its associated probability'
    for col in range(3):
        if np.random.binomial(1, axis[col]):
            coords[:,col] = np.negative(coords[:,col])
    return coords

def coords_to_volume(coords, v_size, noise_treatment=False):
    'Converts coordinates to binary voxels' # Input is centered on [0,0,0]
    return weights_to_volume(coords=coords, weights=1, v_size=v_size, noise_treatment=noise_treatment)

def weights_to_volume(coords, weights, v_size, noise_treatment=False):
    'Converts coordinates to voxels with weights' # Input is centered on [0,0,0]
    # Initialization
    volume = np.zeros((v_size, v_size, v_size))

    # Translate center
    coords = coords + np.full((coords.shape[0], 3), (v_size-1)/2)

    # Round components
    coords = coords.astype(int)

    # Filter rows with values that are out of the grid
    mask = ((coords >= 0) & (coords < v_size)).all(axis=1)

    # Convert to volume
    volume[tuple(coords[mask].T)] = weights[mask] if type(weights) != int else weights

    # Remove noise
    if noise_treatment == True:
        volume = remove_noise(coords, volume)

    return volume

def precomputed_name(pdb_id, path, type_file, desired_p, weights_name=None, scaling=True):
    'Returns path in string of precomputed file'
    if type_file == 'coords':
        return os.path.join(path, pdb_id.lower() + '_coords_p' + str(desired_p) + '.npy')
    elif type_file == 'weights':
        return os.path.join(path, pdb_id.lower() + '_' + weights_name + '_p' + str(desired_p) + '_scaling' + str(scaling) + '.npy')
    
def get_barycenter(coords):
    'Gets barycenter point of a Nx3 matrix'
    return np.array([np.mean(coords, axis=0)])