import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
import math
from  utils import get_gaussian_filter
from termcolor import colored, cprint




class CNNNormal(nn.Module):
  def __init__(self, nc, num_classes,precision_point,dataset, kernel_type, std=1):
    super(CNNNormal, self).__init__()

    self.precision_point = precision_point
    self.dataset = dataset
    self.kernelType = kernel_type

    self.conv1 = nn.Conv2d(nc, 32, kernel_size=3, padding=1)
    self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc = nn.Linear(256 * 2 * 2, 256 * 2 * 2) #for 32*32 Images
    self.classifier = nn.Linear(256 * 2 * 2, num_classes) #for 32*32 Images
    self.std = std
    self.std0 = std
    self.std1 = std
    self.std2 = std
    self.std3 = std
    self.std4 = std


    self.mean1 = std
    self.mean2 = std
    self.mean3 = std
    self.mean4 = std


    directory = "CSV_Data/"
    precision_point = str(self.precision_point)

    CSVFile = "CNN_CBS_"+ self.dataset+"_Precision_Point_" + precision_point + "_kernel_Type_" + str(self.kernelType)+ ".csv"
    self.cfile = directory + CSVFile
    current_working_dir = os.getcwd()
    filename = os.path.join(current_working_dir, self.cfile)
    print(filename)
    header = ['Data_Std','CBS_Layer_1_Std','CBS_Layer_2_Std','CBS_Layer_3_Std','CBS_Layer_4_Std',
              'Data_Mean','CBS_Layer_1_Mean','CBS_Layer_2_Mean','CBS_Layer_3_Mean','CBS_Layer_4_Mean','CBS_Std']
    with open(filename, mode='w') as write_obj:
      csvwriter = csv.writer(write_obj)
      csvwriter.writerow(header)

    #Data Writing for one iteration
    directory = "CSV_Data/Epoch/"
    CSVFile = "data" + ".csv"
    self.dataFile = directory + CSVFile
    print(self.dataFile)

  def get_new_kernels(self, epoch_count):
      #if epoch_count % 10 == 0:
        #self.std *= 0.925
      #self.kernel0 = get_gaussian_filter(mean=self.mean1, kernel_size=3, sigma=self.std0 / 1, channels=3)
      self.kernel1 = get_gaussian_filter(epoch_number=epoch_count,mean=self.mean1, kernel_size=3, sigma=self.std1 / 1, channels=32)
      self.kernel2 = get_gaussian_filter(epoch_number=epoch_count,mean=self.mean2, kernel_size=3, sigma=self.std2 / 1, channels=64)
      self.kernel3 = get_gaussian_filter(epoch_number=epoch_count,mean=self.mean3, kernel_size=3, sigma=self.std3 / 1, channels=128)
      self.kernel4 = get_gaussian_filter(epoch_number=epoch_count,mean=self.mean4, kernel_size=3, sigma=self.std4 / 1, channels=256)


  def get_std(self,x,x1,x2,x3,x4):
    with torch.no_grad():
      #Initial Image
      y = x.float()
      mean0 = torch.mean(y)
      sd0 = torch.std(y)
      #Layer One
      y1 = x1.float()
      mean1 = torch.mean(y1)
      sd1 = torch.std(y1)
      #Layer Two
      y2= x2.float()
      mean2 = torch.mean(y2)
      sd2 = torch.std(y2)
      #Layer Three
      y3 = x3.float()
      mean3 = torch.mean(y3)
      sd3 = torch.std(y3)
      #Layer Four
      y4 = x4.float()
      mean4 = torch.mean(y4)
      sd4 = torch.std(y4)

    current_working_dir = os.getcwd()
    filename = os.path.join(current_working_dir, self.cfile)
    row = [sd0.item(),sd1.item(),sd2.item(),sd3.item(),sd4.item(),mean0.item(),mean1.item(),mean2.item(),mean3.item(),mean4.item(),self.std0]
    with open(filename, mode='a') as write_obj:
      csvwriter = csv.writer(write_obj)
      csvwriter.writerow(row)

    # print(filename)
    dataFilename = os.path.join(current_working_dir, self.dataFile)
    row = [sd1.item(), sd2.item(), sd3.item(), sd4.item()]
    with open(dataFilename, mode='a') as write_obj:
      csvwriter = csv.writer(write_obj)
      csvwriter.writerow(row)


  def setTrainingModelStatus(self,modelStatus):
    self.modelStatus = modelStatus

  def forward(self, x):

      x1 = self.conv1(x)

      with torch.no_grad():
        y = x1.float()
        self.mean1 = torch.mean(y).item()
        self.std1 = torch.std(y).item()

      x1 = self.kernel1(x1)
      x1 = F.relu(self.max1(x1))

      x2 = self.conv2(x1)
      with torch.no_grad():
        y = x2.float()
        self.mean2 = torch.mean(y).item()
        self.std2 = torch.std(y).item()


      x2 = self.kernel2(x2)
      x2 = F.relu(self.max2(x2))

      x3 = self.conv3(x2)
      with torch.no_grad():
        y = x3.float()
        self.mean3 = torch.mean(y).item()
        self.std3 = torch.std(y).item()

      x3 = self.kernel3(x3)
      x3 = F.relu(self.max3(x3))

      x4 = self.conv4(x3)
      with torch.no_grad():
        y = x4.float()
        self.mean4 = torch.mean(y).item()
        self.std4 = torch.std(y).item()

      x4 = self.kernel4(x4)
      x4 = F.relu(self.max4(x4))

      if self.modelStatus:
        self.get_std(x, x1, x2, x3, x4)


      x4 = x4.view(x4.size(0), -1)

      x4 = F.relu(self.fc(x4))

      x4 = self.classifier(x4)

      return x4


class SimpleMLP(nn.Module):
  def __init__(self, num_classes, input_dim):
    super(SimpleMLP, self).__init__()

    self.fc1 = nn.Linear(input_dim, 500)
    self.fc2 = nn.Linear(500, 500)
    self.fc3 = nn.Linear(500, num_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


class OneLayerMLP(nn.Module):
  def __init__(self, num_classes, input_dim):
    super(OneLayerMLP, self).__init__()
    self.fc1 = nn.Linear(input_dim, num_classes)

  def forward(self, x):
    x = self.fc1(x)
    return x
