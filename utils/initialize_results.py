"""Initialize results CSV file."""

import csv

with open('./experiments/results.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow(['MODEL', 'DATASET', 'DISTRIBUTION', 'KERNEL TYPE', 'BEST ACCURACY', '@ EPOCH'])

    for model in ['Normal CNN', 'Resnet 18', 'VGG 16']:
        for dataset in ['Cifar 10', 'Cifar 100', 'MNIST', 'SVHN']:
            for distro in ['None', 'Cauchy', 'Gaussian', 'Gumbel', 'Laplace', 'Poisson']:
                for i in range(1, 15):
                    writer.writerow([model, dataset, distro, i, '--', '--'])