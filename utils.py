# Read csv and returns a matrix.
def read_csv(file_name, num_inputs):
    X = []
    y = []
    with open(file_name, 'r') as file:
        for data in file:
            split_data = data.split(',')
            X.append([float(n) for n in split_data[:num_inputs]])
            y.append([float(n) for n in split_data[num_inputs:]])
    return (X,y)