import torch

def variable_one_hot(shape):
    tensor = torch.zeros(shape, dtype = torch.float32)
    tensor[..., 0] = 1
    return tensor

if __name__ == '__main__':
    shape = [2,3,4]
    one_hot_tensor = variable_one_hot(shape)
    print(one_hot_tensor)
