"""Initialize results CSV file."""

import csv

with open('./experiments/results.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow(['MODEL', 'DATASET', 'DISTRIBUTION', 'KERNEL TYPE', 'BEST ACCURACY', '@ EPOCH'])

    for model in ['normal', 'resnet', 'vgg']:
        for dataset in ['cifar10', 'cifar100', 'mnist', 'svhn']:
            for distro in ['none', 'cauchy', 'gaussian', 'gumbel', 'laplace', 'poisson']:
                for i in range(1, 15):
                    writer.writerow([model, dataset, distro, i, '--', '--'])