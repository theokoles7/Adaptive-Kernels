"""Operations pertaining to dataset selection and configuration."""

import argparse, os, random, torch.utils.data as data, typing
from torchvision import transforms, datasets

def get_data(args: argparse.NameSpace) -> typing.Tuple[data.DataLoader, data.DataLoader, argparse.NameSpace]:
    """Facilitate fetching/loading of data sets.

    Args:
            args (argparse.NameSpace): Parsed command line arguments

    Raises:
            NotImplementedError: If args.dataset points to a data set that is not yet supported

    Returns:
            typing.Tuple[data.DataLoader, data.DataLoader, argparse.NameSpace]: Test & train data loaders and relay of command line arguments
    """
    # Use arguments passed to determine dataset.
    match args.dataset:

        case 'caltech':
            """(https://data.caltech.edu/records/mzrjq-6wc02) Pictures of objects belonging to 101 categories. About 40 to 800 images per category. 
            Most categories have about 50 images. Collected in September 2003 by Fei-Fei Li, Marco Andreetto, and Marc'Aurelio Ranzato. The size 
            of each image is roughly 300 x 200 pixels.
            """
            transform =         transforms.Compose([transforms.CenterCrop(128),transforms.Resize(64),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,)) ])
            test_transform =    transforms.Compose([transforms.CenterCrop(128),transforms.Resize(64),transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3, 1, 1))           ])

            train_data =    datasets.Caltech101(root=args.data,download=False,transform=transform       )
            test_data =     datasets.Caltech101(root=args.data,download=False,transform=test_transform  )

            args.num_classes = 101

        case 'cifar10':
            """The CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 32x32 colour images in 10 classes, with 6000 images 
            per class. There are 50000 training images and 10000 test images. 
            """
            transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])

            train_data =    datasets.CIFAR10(root=args.data,download=True,train=True,transform=transform    )
            test_data =     datasets.CIFAR10(root=args.data,download=True,train=False,transform=transform   )

            args.num_classes =  10
            args.in_dim =       3

        case 'cifar100':
            """This dataset is just like the CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html), except it has 100 
            classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 
            are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the 
            superclass to which it belongs). 
            """
            transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])
    
            train_data =    datasets.CIFAR100(root=args.data,download=True,train=True,transform=transform   )
            test_data =     datasets.CIFAR100(root=args.data,download=True,train=False,transform=transform  )

            args.num_classes =  100
            args.in_dim =       3

        case 'imagenet':
            """ImageNet (https://image-net.org/) is an image database organized according to the WordNet hierarchy (currently only the nouns), in 
            which each node of the hierarchy is depicted by hundreds and thousands of images. The project has been instrumental in advancing computer 
            vision and deep learning research. The data is available for free to researchers for non-commercial use.
            """
            transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])

            train_data =    datasets.ImageFolder(os.path.join(args.data, 'tiny-imagenet-200', 'train'),transform=transform  )
            test_data =     datasets.ImageFolder(os.path.join(args.data, 'tiny-imagenet-200', 'val'),transform=transform    )

            args.num_classes =  200
            args.in_dim =       3
           
        case 'mnist':
            """The MNIST (http://yann.lecun.com/exdb/mnist/) database of handwritten digits, available from this page, has a training set of 
            60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been 
            size-normalized and centered in a fixed-size image. 
            """
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    
            train_data =    datasets.MNIST(root=args.data,download=True,train=True,transform=transform  )
            test_data =     datasets.MNIST(root=args.data,download=True,train=False,transform=transform )

            args.num_classes =  10
            args.in_dim =       28*28 

        case 'svhn':
            """SVHN (http://ufldl.stanford.edu/housenumbers/) is a real-world image dataset for developing machine learning and object recognition 
            algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of 
            small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly 
            harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in 
            Google Street View images.
            """    
            transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))])
    
            train_data =    datasets.SVHN(root=args.data,download=True,split='train',transform=transform)
            test_data =     datasets.SVHN(root=args.data,download=True,split='test',transform=transform)

            args.num_classes =  10
            args.in_dim =       3

        case _:
            raise NotImplementedError(f"Support for {args.dataset} has not been implemented.")
              
    train_loader = data.DataLoader(
        train_data, 
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
        drop_last=True,
        sampler=(
            data.sampler.SubsetRandomSampler(
                random.sample(
                    [i for i in range(len(train_data))], 
                    int(args.percentage*len(train_data)/100))) if args.ssl else None
            )
    )


    test_loader = data.DataLoader(
        test_data, 
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        drop_last=False,
    )

    return train_loader, test_loader, args

