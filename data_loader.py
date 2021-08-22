import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import  Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
data_path='D:/Automatic Vehicle Segmentation/bdd100k_images_10k/bdd100k/images/10k/train'
label_path= '/Automatic Vehicle Segmentation/bdd100k_sem_seg_labels_trainval/bdd100k/labels/sem_seg/masks/train_labels_images'
class Custom_data_loader_for_Seg(Dataset):
    def __init__(self, path, label_path, labelMapping=None):
        self.path=path
        self.output_path=label_path
        self.label_path=label_path
        self.all_images=sorted(os.listdir(path))
        """
            
        """
        if labelMapping is None:
            self.labelMapping=set()
            for labelI in os.listdir(label_path):
               self.labelMapping.update(set([int(i.split('.')[0]) for i in os.listdir(label_path + '/' + labelI)]))
            self.labelMapping=sorted(list(self.labelMapping))
        else:
            self.labelMapping=labelMapping

        self.imageshape=[576,1024]
        self.aspect_ratio=self.imageshape[1]/self.imageshape[0]

        """
            The aspect_resize image takes image and resize the images into 576*1024 and converts the target labels into 
            gray_scale images
        """


    def aspect_resize(self, image, convert=None):
        aspect= image.shape[1]/image.shape[0]
        if aspect==self.aspect_ratio:
            image = np.asarray(image)
            image=Image.fromarray(image).resize((self.imageshape[0], self.imageshape[1]))
            if convert is not None:
                image.convert(convert)
            return np.array(image)
        else:
            print('Aspect resize not done')
            exit(0)


    def __getitem__(self, item):
        image=plt.imread(self.path + '/' + self.all_images[item])
        image=self.aspect_resize(image)
        target=np.zeros([image.shape[0], image.shape[1]]) #array of zeros(576*1024)
        labeldir=self.label_path + '/' + self.all_images[item][:-4]
        for labelI in sorted(os.listdir(labeldir)):
            targetI=cv2.imread(label_path + '/' + labelI)
            targetI=self.aspect_resize(targetI, 'L')
            target[targetI==255]=self.labelMapping.index(int(labelI.split('.')[0]))
        return image.transpose(2,0,1).astype(np.float32)/255, target.astype(np.int64)



    def __len__(self):
        return  len(self.all_images)

if __name__ == '__main__':
    dataset=Custom_data_loader_for_Seg(data_path, label_path)
    trainloader=DataLoader(dataset, num_workers=0, batch_size=4, shuffle=True)
    progress_bar=tqdm(trainloader)
    setOf=set()
    for data, targetI in progress_bar:
        setOf.update(set(np.unique(targetI).tolist()))
        progress_bar.set_descriptiom(str(setOf))





