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

# Apply minmax normalization to a matrix
def minmax_normalize(data):
    cols = len(data[0])
    new_data = data.copy()
    max_data = []
    min_data = []
    for col in range(cols):
        col_values = [row[col] for row in data]
        col_max = max(col_values)
        col_min = min(col_values)
        max_data.append(col_max)
        min_data.append(col_min)
        for row in new_data:
            row[col] = (row[col] - col_min) / (col_max - col_min)
    return (new_data, max_data, min_data)