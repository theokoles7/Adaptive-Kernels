"""Basic CNN model."""

import torch, torch.nn as nn, torch.nn.functional as F

class NormalCNN(nn.Module):
    """Basic CNN model"""

    def __init__(self, dataset_name: str, channels_in: int, channels_out: int, std: int = 1):
        """Initialize Normal CNN model.

        Args:
            dataset_name (str): Target dataset name
            channels_in (int): Input channels
            channels_out (int): Output channels (number of classes in dataset)
            std (int, optional): Standard deviation?. Defaults to 1.
        """
        super(NormalCNN, self).__init__()

        # Concolving layers
        self.conv1 = nn.Conv2d(channels_in,  32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(         32,  64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(         64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(        128, 256, kernel_size=3, padding=1)

        # Max pooling layers
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc = nn.Linear(256 * 2 * 2, 256 * 2 * 2)

        # Classifier
        self.classifier = nn.Linear(256 * 2 * 2, channels_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Feed input through network and return output.

        Args:
            X (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x1 = F.relu(self.max1(self.conv1(X)))
        x2 = F.relu(self.max2(self.conv2(x1)))
        x3 = F.relu(self.max3(self.conv3(x2)))
        x4 = F.relu(self.max4(self.conv4(x3)))

        if self.modelStatus:
            self.get_std(X, x1, x2, x3, x4)

        return self.classifier(F.relu(self.fc(x4.view(x4.size(0), -1))))

        # directory = "CSV_Data/"
        # precision_point = str(self.precision_point)
        # CSVFile = "CNN_"+ self.dataset+"_Precision_Point_" + precision_point + ".csv"
        # self.cfile = directory + CSVFile
        # current_working_dir = os.getcwd()
        # filename = os.path.join(current_working_dir, self.cfile)
        # print(filename)
        # header = ['Data_Std', 'CNN_Layer_1_Std', 'CNN_Layer_2_Std', 'CNN_Layer_3_Std', 'CNN_Layer_4_Std',
        #         'Data_Mean', 'CNN_Layer_1_Mean', 'CNN_Layer_2_Mean', 'CNN_Layer_3_Mean','CNN_Layer_4_Mean']
        # with open(filename, mode='w') as write_obj:
        # csvwriter = csv.writer(write_obj)
        # csvwriter.writerow(header)


        #Data Writing for one iteration
        directory = "CSV_Data/Epoch/"
        CSVFile = "data" + ".csv"
        self.dataFile = directory + CSVFile
        print(self.dataFile)

    def get_new_kernels(self, epoch_count):
        if epoch_count % 10 == 0:
        self.std *= 0.925
        self.kernel0 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=3)
        self.kernel1 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=32)
        self.kernel2 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=64)
        self.kernel3 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=128)
        self.kernel4 = get_gaussian_filter(kernel_size=3, sigma=self.std / 1, channels=256)

    def record_std(self, x, x1, x2, x3, x4):

        with torch.no_grad():
            # Initial Image
            y = x.float()
            mean0 = torch.mean(y)
            sd0 = torch.std(y)

            # Layer One
            y1 = x1.float()
            mean1 = torch.mean(y1)
            sd1 = torch.std(y1)

            # Layer Two
            y2 = x2.float()
            mean2 = torch.mean(y2)
            sd2 = torch.std(y2)

            # Layer Three
            y3 = x3.float()
            mean3 = torch.mean(y3)
            sd3 = torch.std(y3)

            # Layer Four
            y4 = x4.float()
            mean4 = torch.mean(y4)
            sd4 = torch.std(y4)

        current_working_dir = os.getcwd()
        filename = os.path.join(current_working_dir, self.cfile)
        row = [sd0.item(),sd1.item(), sd2.item(), sd3.item(), sd4.item(),
            mean0.item(), mean1.item(),mean2.item(), mean3.item(), mean4.item()]
        with open(filename, mode='a') as write_obj:
        csvwriter = csv.writer(write_obj)
        csvwriter.writerow(row)



        #print(filename)
        dataFilename = os.path.join(current_working_dir, self.dataFile)
        row = [sd1.item(), sd2.item(), sd3.item(), sd4.item()]
        with open(dataFilename, mode='a') as write_obj:
        csvwriter = csv.writer(write_obj)
        csvwriter.writerow(row)

    def setTrainingModelStatus(self,modelStatus):
        self.modelStatus = modelStatus