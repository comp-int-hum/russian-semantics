import torch


if __name__ == '__main__':
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)