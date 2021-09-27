import torch
from torch.utils.data import  DataLoader
from mode_training import AutoEncoder
from data_loader import Custom_data_loader_for_Seg
from torch.utils.tensorboard import SummaryWriter
data_path = 'D:/Automatic Vehicle Segmentation/bdd100k_images_10k/bdd100k/images/10k/train'
label_path = 'D:/Automatic Vehicle Segmentation/bdd100k_sem_seg_labels_trainval/bdd100k/labels/sem_seg/masks/train_labels_images'


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device available', {device})
    num_of_epochs = 10
    train_loss_list = []
    valid_loss_list = []
    torch.manual_seed(0)
    dataset = Custom_data_loader_for_Seg(data_path, label_path)
    train_loader = DataLoader(dataset, num_workers=8, batch_size=4, shuffle=True)
    model = AutoEncoder()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epochs in range(num_of_epochs):
        #train_loss = 0.0
        torch.manual_seed(0)
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader):

            optimizer.zero_grad()
            ouput=model(data)
            loss=criterion(ouput, label)
            loss.backward()
            optimizer.step()
            loss.item()
            #train_loss = train_loss / len(train_loader.dataset)
            #train_loss_list.append(train_loss)



            if batch_idx%100 == 0:
                print(f"train_epoch:{epochs}, {batch_idx*len(data)}, {100 * batch_idx / len(train_loader):.0f}, Loss:{ loss.item() :.4f}")
        # calculate average losses

    torch.save(model.state_dict(), "image_seg.pt")

if __name__ == '__main__':
    main()






