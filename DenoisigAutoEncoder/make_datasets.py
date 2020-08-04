from os import mkdir, listdir
from os.path import exists, join, splitext
import numpy as np

if __name__ == '__main__':
    data_input = '../data/Ready'
    data_output = '../data/Encoded_dae'
    if not exists(data_output):
        mkdir(data_output)

    for data_file in listdir(data_input):
        input_path = join(data_input, data_file)
        file_name, _ = splitext(data_file)
        print(file_name)
        motions_path = join('Encoded_dae', file_name + '.npy')
        audio = np.load(input_path)['X']
        motions = np.load(motions_path)

        np.savez(join(data_output, data_file), X=audio, Y=motions)