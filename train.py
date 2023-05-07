from torchvision.models import resnet50
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim, nn
from tqdm import tqdm


model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset, validation_dataset = MNIST(root='./data', download=True, train=True, transform=transform), MNIST(root='./data', download=True, train=False, transform=transform)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True), DataLoader(validation_dataset, batch_size=32, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_num = 10

for epoch in tqdm(range(epoch_num)):
    running_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100}')
            running_loss = 0.0