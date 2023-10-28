"""Drive Hi-Lo application."""

import datetime, logging, os

import pandas as pd, matplotlib.pyplot as plt, torch, torch.nn.functional as F, torch.optim as optim, traceback
from sklearn.metrics import accuracy_score
from termcolor       import colored
from torchvision     import transforms
from tqdm            import tqdm

from utils import ARGS, LOGGER
from utils.banner import *

from datasets   import get_dataset
from models     import get_model

if __name__ == '__main__':
    """Drive operations of HI/LO Machine Learning applicaiton."""
    try:
        # Turn off other module loggers
        # for _ in logging.root.manager.loggerDict: logging.getLogger(_).setLevel(logging.CRITICAL)

        # Print lab banner
        LOGGER.info(lab_banner + arg_banner)

        # Ensure output directory exists
        output_dir = f"{ARGS.output_path}/{ARGS.model}/{ARGS.distribution}{f'/{ARGS.kernel_type}' if ARGS.distribution else ''}/{ARGS.dataset}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        LOGGER.info(f"Output directory: {output_dir}")

        # Fetch dataset
        dataset = get_dataset(ARGS.dataset, ARGS.dataset_path, ARGS.batch_size)
        train, test = dataset.get_loaders()

        if ARGS.debug: LOGGER.debug(f"DATASET:\n{dataset}\nTRAIN LOADER:\n{vars(train)}\nTEST LOADER:\n{vars(test)}")

        # Fetch model
        model = get_model(channels_in=dataset.channels_in, channels_out=dataset.num_classes, dim=dataset.dim)
        if ARGS.debug: LOGGER.debug(f"MODEL:\n{model}")

        # Run model from CUDA if available
        if torch.cuda.is_available(): model = model.cuda()
        LOGGER.info(f"Using device: {torch.cuda.get_device_name()}")

        # Initialize optimizer and decay parameters
        optimizer = optim.SGD(model.parameters(), lr=ARGS.learning_rate, weight_decay=5e-4, momentum=0.9)
        decay_int = (50 if ARGS.model == 'vgg' else 30) # Learning rate decay interval
        decay_lmt = (decay_int * 3) + 1                 # Learning rate decay limit

        # Define loss function
        loss_func = F.cross_entropy

        # TRAINING ================================================================================
        LOGGER.info("Commencing training phase...")

        # Initialize metrics and report
        best_epoch = best_acc = i = 0
        training_report = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Loss'])
        accs, losses = [], []

        for epoch in range(1, ARGS.epochs + 1):
            LOGGER.info(f"BEGIN EPOCH {epoch} =========================================================================")

            # Place model in training mode
            model.train()

            epoch_best_acc = 0

            # Update model kernels
            if ARGS.distribution:
                model.update_kernels(epoch)
                if torch.cuda.is_available(): model = model.cuda()

            # If ARGS.epoch_limit is reached, end training
            if epoch - 1 == ARGS.epoch_limit:
                LOGGER.info(f"Epoch limit of {ARGS.epoch_limit} reached. Ending training...")
                break

            # Administer learning rate decay
            if (epoch % decay_int == 0) and (epoch < decay_lmt):
                for param in optimizer.param_groups:
                    param['lr'] /= 10

            with tqdm(total=len(train)+len(test), desc=f"Epoch {epoch}/{ARGS.epochs}", leave=True, colour="magenta") as pbar:

                # For image:label pairs in train data...
                pbar.set_postfix(status=colored("Training", "cyan"))

                for images, labels in train:
                    if ARGS.debug: LOGGER.debug(f"IMAGE {images}, LABEL {labels}")
                    # plt.imshow(images[0].permute(1, 2, 0)) # <----- For viewing images one by one
                    # plt.title(labels[0])
                    # plt.show()

                    # Load to CUDA if possible
                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()

                    # Make predictions and calculate loss
                    predictions = model(images)
                    loss = loss_func(predictions, labels)

                    # Back propogation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update iterators
                    pbar.update(1)

        # VALIDATION ==============================================================================
                pbar.set_postfix(status=colored("Validating", "yellow"))
                model.eval()

                # Initialize metrics
                total = correct = 0

                # For image:label pairs in test...
                for images, labels in test:

                    # Load to CUDA if available
                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()

                    with torch.no_grad():
                        predictions = model(images)
                        predictions = torch.argmax(predictions, dim=1).cpu().numpy()

                        correct += accuracy_score(labels.cpu(), predictions, normalize=False)
                        total   += images.size(0)

                        acc = round((correct / total) * 100, 2)
                        if acc > epoch_best_acc:
                            epoch_best_acc = acc

                    # Update iterators
                    pbar.update(1)

                # Update and communicate final accuracy for epoch
                if epoch_best_acc > best_acc:
                    best_acc   = epoch_best_acc
                    best_epoch = epoch

                pbar.set_postfix(status=f"Accuracy: {colored(round(epoch_best_acc, 1), ('green' if epoch_best_acc == best_acc else 'red'))}%")

                accs.append(epoch_best_acc)
                losses.append(loss.item())
                training_report.loc[epoch-1] = [int(epoch), accs[-1], losses[-1]]

            LOGGER.info(f"END EPOCH: {epoch:>3} | ACCURACY: {accs[-1]} | LOSS: {losses[-1]}")

        # TESTING =================================================================================
        LOGGER.info("Commencing testing phase...")

        with tqdm(total=len(test), desc=f"{colored('Testing', 'magenta')}", leave=True, colour='cyan') as pbar:

            pbar.set_postfix(status="Testing")

            # Initialize metrics
            total = correct = 0

            # For image:label pairs in test data...
            for images, labels in test:

                # Load to CUDA if available
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                with torch.no_grad():
                    predictions = model(images)
                    predictions = torch.argmax(predictions, dim=1).cpu().numpy()

                    correct += accuracy_score(labels.cpu(), predictions, normalize=False)
                    total +=   images.size(0)

                    acc = (correct / total) * 100
                    if acc > best_acc:
                        best_acc   = acc
                        best_epoch = epoch

                # Update iterator
                pbar.update(1)

            # Communicate final accuracy
            pbar.set_postfix(status=f"Accuracy: {round(acc, 1)}%")

        training_report.loc[ARGS.epochs + 1] = ['Test', acc, '']
        training_report.loc[ARGS.epochs + 2] = ['Best', best_acc, f'@ epoch {best_epoch}']

        LOGGER.info(f"Highest accuracy of {best_acc}% @ epoch {best_epoch}")

        # Record results
        LOGGER.info("Recording results")
        results_file = pd.read_csv('experiments/results.csv')
        results_file.loc[
            (results_file['MODEL']==ARGS.model) & 
            (results_file['DATASET']==ARGS.dataset) & 
            (results_file['DISTRIBUTION']==ARGS.distribution) & 
            (results_file['KERNEL TYPE']==ARGS.kernel_type), 
            ['BEST ACCURACY', '@ EPOCH']] = best_acc, best_epoch
        results_file.to_csv('experiments/results.csv', index=False)

        # Save performance report and graph
        plt.plot(accs, label='Accuracy (%)')
        plt.plot(losses, label='loss')
        plt.title(
            f"MODEL: {ARGS.model} | DATASET: {ARGS.dataset}"
            f"\nDISTRIBUTION: {ARGS.distribution} | KERNEL TYPE: {ARGS.kernel_type}"
        )
        plt.xlabel('Epoch')
        plt.legend()

        plt.savefig(f"{output_dir}/training_graph.jpg")
        LOGGER.info("Saved plotted graph to {f'{output_dir}/training_graph.jpg'}")

        training_report.to_csv(f"{output_dir}/training_report.csv", index=False)
        LOGGER.info(f"Saved training report to {output_dir}/training_report.csv")

        # Save model data
        model.to_csv(f"{output_dir}/model.csv")
        LOGGER.info(f"Saved model data to {output_dir}/model.csv")

    except KeyboardInterrupt:
        LOGGER.info(f"Keyboard interrupt detected. Aborting operations...")
        exit(0)

    except Exception as e:
        LOGGER.error(f"An error occurred: {e}")
        traceback.print_exc()