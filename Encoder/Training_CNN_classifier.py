import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms.functional as Fun
import matplotlib.pyplot as plt
import os
from torchsummary import summary


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return Fun.pad(image, padding, 0, 'constant')


# train and test data directory
data_dir = "/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/dataset_pulito/train_test_class/fold_0/rgb/train_aug"
test_data_dir = "/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/dataset_pulito/train_test_class/fold_0/rgb/test"
outputFolder = "/home/sealab-ws/PycharmProjects/Apicella_PyCharm/Visual/Encoder/OUTPUTS_class"
transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the train and test data
dataset = ImageFolder(data_dir, transform=transform)
test_dataset = ImageFolder(test_data_dir, transform=transform)

batch_size = 20
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size

train_data, val_data = random_split(dataset, [train_size, val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")

# load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle=True)
val_dl = DataLoader(val_data, batch_size * 2)

import torch.nn as nn
import torch.nn.functional as F


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def test_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'test_loss': loss.detach(), 'test_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def test_epoch_end(self, outputs):
        batch_losses = [x['test_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['test_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    def epoch_end_test(self, epoch, result):
        print("Epoch [{}], , test_loss: {:.4f}, test_acc: {:.4f}".format(
            epoch, result['test_loss'], result['test_acc']))


class NaturalSceneClassification(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(51200, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, xb):
        return self.network(xb)


class MobileNetV1(ImageClassificationBase):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.Dropout(0.2),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Dropout(0.2),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.Dropout(0.2),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            nn.AvgPool2d(2, 2),
            conv_dw(32, 64, 1),
            nn.AvgPool2d(2, 2),
            conv_dw(64, 128, 2),
            nn.AvgPool2d(2, 2),
            conv_dw(128, 128, 1),
            # nn.AvgPool2d(2, 2),
            # conv_dw(128, 256, 2),
            # conv_dw(256, 256, 1),
            # conv_dw(256, 512, 2),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 1024, 2),
            # conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc1 = nn.Linear(128, n_classes)
        self.fc2 = nn.Linear(9, n_classes)

    def forward(self, x): #, depthAndImagesRatios):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc1(x)
        # x = nn.ReLU()(x)
        # x = torch.cat((x, depthAndImagesRatios), dim=1)
        # x = self.fc2(x)

        return x


def get_default_device():
    """ Set Device to GPU or CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    "Move data to the device"
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """ Wrap a dataloader to move data to a device """

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """ Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """ Number of batches """
        return len(self.dl)


device = get_default_device()

num_epochs = 5
opt_func = torch.optim.Adam
lr = 0.001  # fitting the model on training data and record the result after each epoch
# load the into GPU
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
model = to_device(MobileNetV1(3, 3), device)
summary(model, input_size=(3, 224, 224))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


@torch.no_grad()
def evaluate_test(model, val_loader):
    model.eval()
    outputs = [model.test_step(batch) for batch in val_loader]
    return model.test_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):

        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        torch.save(model.state_dict(), os.path.join(outputFolder, 'CNN_{}.torch'.format(epoch)))

    return history


history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig("Accuracy")


plot_accuracies(history)


def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig("Loss")


plot_losses(history)

for epoch in range(num_epochs):
    history = []
    model.load_state_dict(torch.load(os.path.join(outputFolder, 'CNN_{}.torch'.format(epoch))))
    # Apply the model on test dataset and Get the results
    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size), device)
    result = evaluate_test(model, test_loader)
    model.epoch_end_test(epoch, result)
    history.append(result)
