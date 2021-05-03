from torch.utils.data.dataloader import default_collate

'''
Code was taken from https://github.com/uvipen/Yolo-v2-pytorch
'''

def custom_collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    return items