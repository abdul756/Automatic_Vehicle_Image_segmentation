import numpy as np
import os
from torch.utils.data import  Dataset
import matplotlib.pyplot as plt
data_path='D:/Automatic Vehicle Segmentation/bdd100k_images_10k/bdd100k/images/10k/train'
label_path='D:/Automatic Vehicle Segmentation/bdd100k_sem_seg_labels_trainval/bdd100k/labels/sem_seg/masks/train'
output_path='D:/Automatic Vehicle Segmentation/bdd100k_sem_seg_labels_trainval/bdd100k/labels/sem_seg/masks'
class Custom_data_loader_for_Seg(Dataset):
    def __init__(self,path,output_path, label_path):  # initialisation of variables
        self.path=path
        self.output_path=output_path
        self.label_path=label_path
        self.all_images=sorted(os.listdir(path))
        self.imageshape=[576,1024]
        self.aspect_ratio=self.imageshape[1]/self.imageshape[0]  #1024/255

    def split_into_subimages(self, labels, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        unique_labels=np.unique(labels)
        for i in unique_labels:
            plt.imsave(output_dir + '/' + str(i) + '.png', (labels==i).astype(np.uint8)*255, cmap='gray')

    def __getitem__(self, item):
        '''
        getitem is a method in which item as acts an indexer
        It iterates through all the images in labels/masks/train folder and labels are converted to unique labels ranging from,
        1 , 2 ....
        :param item:
        :return:
        '''

        labels=(plt.imread(self.label_path + '/' + self.all_images[item][:-3] + 'png')*255).astype(np.uint8)
        self.split_into_subimages(labels, self.output_path + '/'+ self.all_images[item][:-4])
        return  self.output_path + '/' + self.all_images[item][:-4]

    def __len__(self):
        return  len(self.all_images)

if __name__ == '__main__':
        customDataLoaderObject = DataLoader(
        Custom_data_loader_for_Seg(data_path,output_path,label_path),
        batch_size=1,
        num_workers=0,
        shuffle=True
    )

    for images in enumerate(customDataLoaderObject):
        print(images)





