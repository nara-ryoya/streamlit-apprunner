from torchvision.models import resnet50
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim, nn
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = resnet50(pretrained=True)
# for p in model.parameters():
#     p.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset, validation_dataset = MNIST(root='./data', download=True, train=True, transform=transform), MNIST(root='./data', download=True, train=False, transform=transform)

train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True), DataLoader(validation_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_num = 10

for epoch in tqdm(range(epoch_num)):
    print(f'Epoch {epoch + 1}')
    running_loss = 0.0
    model.train()
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print(f'[{epoch + 1}, {i + 1} / {len(train_dataloader)}] loss: {running_loss / 100}')
            running_loss = 0.0

    with torch.no_grad():
        model.eval()
        all_correct_sum = 0
        all_sample_num = 0
        for i, (inputs, labels) in tqdm(enumerate(test_dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            correct_sum = torch.sum(torch.argmax(outputs, dim=1) == labels)
            all_correct_sum += correct_sum
            all_sample_num += len(labels)
        print(f'Accuracy: {all_correct_sum / all_sample_num}')

    torch.save(model.state_dict(), f'./models/model_{epoch + 1}_acc_{(all_correct_sum / all_sample_num):.2}.pth')