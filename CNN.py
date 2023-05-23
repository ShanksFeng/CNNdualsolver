import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import onnx
import onnxruntime
from torch.optim import lr_scheduler
import torch.nn.functional as F
import glob
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import random_split
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
import pytorch_ssim
def read_multiple_files(pattern):
    file_names = glob.glob(pattern)
    data = []
    for file_name in file_names:
        file_data = read_data(file_name)
        data.extend(file_data)
    return data

def read_data(file_name):
    data = []
    input_data = [None]*3
    output_data = None
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            if line.startswith("location"):
                if None not in input_data and output_data is not None:
                    data.append((input_data, output_data))  # Store the complete data block
                    input_data = [None]*3  # Reset input_data for the next data block
                    output_data = None
                x, y = map(float, line[len("location :"):].split())
                input_data[0] = [x, y]
            elif line.startswith("mach_number"):
                mach_number = float(line[len("mach_number :"):])
                input_data[0].append(mach_number)
            elif line.startswith("attack_angle"):
                attack_angle = float(line[len("attack_angle :"):])
                input_data[0].append(attack_angle)
            elif line.startswith("cellSquare"):
                cellSquare = float(line[len("cellSquare :"):])
                input_data[0].append(cellSquare)
            elif line.startswith("SolutionAverage"):
                cellAverage = list(map(float, line[len("SolutionAverage :"):].split()))
                input_data[1] = cellAverage
            elif line.startswith("local_residual"):
                local_residual = list(map(float, line[len("local_residual :"):].split()))
                input_data[2] = local_residual
                #print(f'input_data after processing line: {input_data}')  # Debug line
            elif line.startswith("adjoint_solution"):
                adjoint_solution = list(map(float, line[len("adjoint_solution :"):].split()))
                output_data = adjoint_solution
                #print(f'out_data after processing line: {adjoint_solution}')

    # After reading the entire file, check again if we have a complete data block to save
    if None not in input_data and output_data is not None:
        data.append((input_data, output_data))

    return data

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data, output_data = self.data[idx]
        # Get the fifth element of the first feature
        fill_value = input_data[0][4] if len(input_data[0]) >= 5 else 0
        # Fill the other features with the fill_value if necessary
        for i in range(1, len(input_data)):
            if len(input_data[i]) < 5:
                input_data[i] = input_data[i] + [fill_value]
        # Convert to tensor and adjust dimensions
        input_data = torch.FloatTensor(input_data).unsqueeze(1) # Adding channel dimension
        output_data = torch.FloatTensor(output_data)
        return input_data, output_data


# Read multiple data files
train_data = read_multiple_files('Train*.dat')
#print('Number of training samples:', len(train_data))
#print('Shape of the first input tensor:', len(train_data[0][0][0]))
#print('Shape of the first output tensor:', len(train_data[0][1]))

# Split the data into training set and test set
train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Convert the lists of data to CustomDataset objects
train_data = CustomDataset(train_data)
test_data = CustomDataset(test_data)

# Print dimension information
print("Height:", train_data[0][0].shape[1])
print("Width:", train_data[0][0].shape[2])
print("Channels:", train_data[0][0].shape[0])
train_loader = DataLoader(train_data, batch_size=65536, shuffle=True)
test_loader = DataLoader(test_data, batch_size=65536, shuffle=False)


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, x):
        energy = self.V(torch.tanh(self.W(x)))  
        attention_weights = F.softmax(energy, dim=1)  
        return attention_weights


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.downsample(residual)
        out += residual
        out = F.elu(out)
        return out


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()
        
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3 = self.branch3x3(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Encoder layers
        self.enc1 = nn.Conv2d(3, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.enc2 = Inception(64, 128)  
        self.enc3 = Inception(128*4, 256)
        self.enc4 = ResBlock(256*4, 512, stride=2, kernel_size=3) 
        self.enc5 = ResBlock(512, 1024, kernel_size=3)
        self.enc6 = ResBlock(1024, 2048, kernel_size=3)

        # Skip connection adjustment
        self.skip_adjust = nn.Conv2d(512, 2048, kernel_size=1)

        # Decoder layers
        self.dec1 = ResBlock(2048, 1024, kernel_size=3)
        self.dec2 = Inception(1024, 512) 
        self.dec3 = ResBlock(512*4, 256, kernel_size=3) 
        self.dec4 = Inception(256, 128)  

        # Global average pooling and output layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128*4, 64)  # Adjusted for Inception output
        # Separate output layers
        self.fc2_1 = nn.Linear(64, 1)
        self.fc2_2 = nn.Linear(64, 1)
        self.fc2_3 = nn.Linear(64, 1)
        self.fc2_4 = nn.Linear(64, 1)


    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x4 = self.enc4(x)
        x = self.enc5(x4)
        x6 = self.enc6(x)

        # Adjust x4 to match x6 dimensions
        x4_adjusted = self.skip_adjust(x4)

        # Decoder
        x = self.dec1(x6 + x4_adjusted)  
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        # Global average pooling and output layer
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = F.elu(self.fc1(x))
        
        # Separate output layers
        out1 = self.fc2_1(out)
        out2 = self.fc2_2(out)
        out3 = self.fc2_3(out)
        out4 = self.fc2_4(out)


        return out1, out2, out3, out4

k_folds = 5
clip_value = 5

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")


len_train_data = len(train_data)
len_val_data = len_train_data // k_folds

indices = torch.randperm(len_train_data).tolist()
datasets = [Subset(train_data, indices[i*len_val_data: (i+1)*len_val_data]) for i in range(k_folds)]


for fold in range(k_folds):
    
    model = Net()
    model.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0002)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)

    print(f'FOLD {fold}')
    print('--------------------------------')

    val_data = datasets[fold]
    train_subsets = [datasets[i] for i in range(k_folds) if i != fold]
    train_data = ConcatDataset(train_subsets)

    train_loader = DataLoader(train_data, batch_size=65536, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=65536, shuffle=False)

    clip_value = 5
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            output1, output2, output3, output4 = model(inputs)

            # Split the targets into separate variables
            target1, target2, target3, target4 = torch.split(targets, 1, dim=1)

            # Compute the loss for each output
            loss1 = criterion(output1, target1)
            loss2 = criterion(output2, target2)
            loss3 = criterion(output3, target3)
            loss4 = criterion(output4, target4)
            relative_error1 = loss1 / (target1 + 1e-10)
            relative_error2 = loss2 / (target2 + 1e-10)
            relative_error3 = loss3 / (target3 + 1e-10)
            relative_error4 = loss4 / (target4 + 1e-10)
            # Sum up the losses
            #loss = loss1 + loss2 + loss3 + loss4
            loss1 = abs(torch.mean(relative_error1))
            loss2 = abs(torch.mean(relative_error2))
            loss3 = abs(torch.mean(relative_error3))
            loss4 = abs(torch.mean(relative_error4))
            loss = loss1 + loss2 + loss3 + loss4
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            running_loss += loss.item()

        running_loss /= len(train_loader)

        model.eval()  
        valid_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output1, output2, output3, output4 = model(inputs)

                
                target1, target2, target3, target4 = torch.split(targets, 1, dim=1)


                loss1 = criterion(output1, target1)
                loss2 = criterion(output2, target2)
                loss3 = criterion(output3, target3)
                loss4 = criterion(output4, target4)
                relative_error1 = loss1 / (target1 + 1e-10)
                relative_error2 = loss2 / (target2 + 1e-10)
                relative_error3 = loss3 / (target3 + 1e-10)
                relative_error4 = loss4 / (target4 + 1e-10)
                loss1 = abs(torch.mean(relative_error1))
                loss2 = abs(torch.mean(relative_error2))
                loss3 = abs(torch.mean(relative_error3))
                loss4 = abs(torch.mean(relative_error4))
                loss = loss1 + loss2 + loss3 + loss4


                valid_loss += loss.item()

        valid_loss /= len(val_loader)

        scheduler.step(valid_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Fold {fold+1}/{k_folds}, Training Loss: {running_loss}, Validation Loss: {valid_loss}")
      
    print('--------------------------------')


# 保存模型为ONNX文件
batch_size, channels, height, width = inputs.shape
dummy_input = torch.randn(1,  channels, height, width).to(device)
model.to(device)
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=['input'], output_names=["out1", "out2", "out3", "out4"])
