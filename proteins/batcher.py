import os
import numpy as np
import pickle

def gen_dataset(dataset='test'):
    #base_path = os.environ.get('PCM_DATA_PATH')
    #assert(base_path != None)

    dataset_fnames = {
        'train': 'data/pdb25-6767-train.release.contactFeatures.pkl', # 5G, >6300 proteins
        'valid': 'data/pdb25-6767-valid.release.contactFeatures.pkl', # 300M, 400 proteins
        'test': 'data/pdb25-test-500.release.contactFeatures.pkl' # 400M, 500 proteins
    }

    dataset_fname = dataset_fnames[dataset]
    full_path = dataset_fname#os.path.join(base_path, dataset_fname)

    with open(full_path, 'rb') as f:
        dataset = pickle.load(f, encoding='latin1')

    return dataset

def get_batch(dataset, batchsize, shuffle=True):
    N = len(dataset)
    while True:
        indices = list(range(N))
        np.random.shuffle(indices)
        for start in range(0, N - batchsize + 1, batchsize):
            idx_batch = indices[start:start+batchsize]
            if len(dataset[idx_batch[0]]['sequence']) <= 50:
                continue
            batch_data = dataset[idx_batch[0]]
            yield batch_data

if __name__ == '__main__':
    dataset = gen_dataset('test') 
    batcher = get_batch(dataset, 4)
    #import ipdb; ipdb.set_trace()  # NOQA