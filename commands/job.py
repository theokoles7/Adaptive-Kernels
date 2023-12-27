"""Job class."""

import datetime, os

import pandas as pd, matplotlib.pyplot as plt, torch, torch.nn.functional as F, torch.optim as optim
from sklearn.metrics    import accuracy_score
from termcolor          import colored
from tqdm               import tqdm

from datasets           import get_dataset
from models             import get_model
from utils              import ARGS, LOGGER

class Job():
    """Job class."""

    # Initialize output directory path
    _output_dir =   f"{ARGS.output_path}/{ARGS.model}/{ARGS.dataset}/{ARGS.distribution}{f'/{ARGS.kernel_type}' if ARGS.distribution else ''}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"

    # Fetch dataset
    _dataset =      get_dataset(ARGS.dataset, ARGS.dataset_path, ARGS.batch_size)
    _train, _test = _dataset.get_loaders()

    # Fetch model
    _model =        get_model(_dataset.channels_in, _dataset.num_classes, _dataset.dim)

    # Initialize optimizer and decay parameters
    _optimizer =    optim.SGD(_model.parameters(), lr=ARGS.learning_rate, weight_decay=5e-4, momentum=0.9)
    _decay_int =    (50 if ARGS.model == 'vgg' else 30)
    _decay_lmt =    (_decay_int * 3) + 1

    # Initialize job statistics
    _best_acc = _best_epoch = _test_acc = 0
    _accs, _losses = [], []
    _training_report = pd.DataFrame(columns = ['Epoch', 'Accuracy', 'Loss'])

    def __init__(self):
        """Initialize job."""
        # Ensure output directory exists
        os.makedirs(self._output_dir, exist_ok=True)
        LOGGER.info(f"Output directory: {self._output_dir}")
        LOGGER.info(f"Using device: {torch.cuda.get_device_name()}")

        # Run model from CUDA if available
        if torch.cuda.is_available(): self._model = self._model.cuda()

        LOGGER.debug(f"DATASET:\n{self._dataset}\nTRAIN LOADER:\n{vars(self._train)}\nTEST LOADER:\n{vars(self._test)}")
        LOGGER.debug(f"MODEL:\n{self._model}")

    def run(self) -> None:
        """Execute job."""
        LOGGER.info(f"Exeucting job: MODEL {ARGS.model} | DATASET {ARGS.dataset} | KERNEL {f'{ARGS.distribution} | TYPE {ARGS.kernel_type}' if ARGS.distribution else 'none'}")

        # Conduct training
        self._train_()

        # Conduct testing
        self._test_()

        # Record results
        self._record_results()

        # Save training report & graph
        self._save_report()
        self._save_graph()

        # Save model data
        self._model.to_csv(f"{self._output_dir}/model_layers.csv")

        LOGGER.info(f"Job complete: MODEL {ARGS.model} | DATASET {ARGS.dataset} | KERNEL {f'{ARGS.distribution} | TYPE {ARGS.kernel_type}' if ARGS.distribution else 'none'}")

    def _record_results(self) -> None:
        """Record job results."""
        LOGGER.info(f"Recording job results in {ARGS.output_path}/results.csv")

        # Open file
        file_out = pd.read_csv(f"{ARGS.output_path}/results.csv")

        # Record job results
        file_out.loc[
            (file_out['MODEL']==str(ARGS.model)) & 
            (file_out['DATASET']==str(ARGS.dataset)) & 
            (file_out['DISTRIBUTION']==str(ARGS.distribution).lower()) & 
            (file_out['KERNEL TYPE']==(str(ARGS.kernel_type) if ARGS.distribution else '--')), 
            ["BEST TRAIN ACCURACY", "AT EPOCH", "TEST ACCURACY"]] = self._best_acc, self._best_epoch, self._test_acc
        
        # Write back
        file_out.to_csv(f"{ARGS.output_path}/results.csv", index=False)

    def _save_graph(self) -> None:
        """Save accuracy graph."""
        LOGGER.info(f"Saving accuracy graph to {self._output_dir}/training_graph.jpg")

        plt.plot(self._accs,    label='Accuracy (%)')
        plt.plot(self._losses,  label='Loss')
        plt.title(
            f"MODEL: {ARGS.model} | DATASET: {ARGS.dataset}"
            f"\nDISTRIBUTION: {ARGS.distribution} | KERNEL TYPE: {ARGS.kernel_type}"
        )
        plt.xlabel('Epoch')
        plt.legend()

        plt.savefig(f"{self._output_dir}/training_graph.jpg")

    def _save_report(self) -> None:
        """Save training report."""
        LOGGER.info(f"Saving training report to {self._output_dir}/training_report.csv")
        self._training_report.to_csv(f"{self._output_dir}/training_report.csv")

    def _train_(self) -> None:
        """Conduct training and validation phase."""
        LOGGER.info("Commencing training phase...")

        for epoch in range(1, ARGS.epochs + 1):

            # Initialize epoch statistics
            epoch_best_acc = 0

            # Set model kernels
            if ARGS.distribution: 
                self._model.set_kernels(epoch)
                if torch.cuda.is_available(): self._model = self._model.cuda()

            # Administer learning rate decay
            if (epoch % self._decay_int == 0) and (epoch < self._decay_lmt):
                for param in self._optimizer.param_groups: param['lr'] /= 10

            with tqdm(total=(len(self._train) + len(self._test)), desc=f"Epoch {epoch}/{ARGS.epochs}", leave=False, colour="magenta") as pbar:
                 
                # TRAIN ---------------------------------------------------------------------------
                pbar.set_postfix(status=colored("Training", "cyan"))

                # Put model into training mode
                self._model.train()

                # For image:label pairs in train data...
                for images, labels in self._train:
                    
                    # Place on GPU if available
                    if torch.cuda.is_available(): images, labels = images.cuda(), labels.cuda()

                    # Make predictions and calculate loss
                    predictions = self._model(images)
                    loss =        F.cross_entropy(predictions, labels)

                    # Back propogation
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    # Update progress bar
                    pbar.update(1)
                
                # VALIDATE ------------------------------------------------------------------------
                pbar.set_postfix(status=colored("Validating", "yellow"))

                # Put model into evaluation mode
                self._model.eval()

                # Initialize validation metrics
                total = correct = 0

                # For image:label pairs in test data...
                for images, labels in self._test:

                    # Place on GPU if available
                    if torch.cuda.is_available(): images, labels = images.cuda(), labels.cuda()

                    # Make predictions and calculate accuracy
                    with torch.no_grad():
                        predictions = self._model(images)
                        predictions = torch.argmax(predictions,dim=1).cpu().numpy()

                        correct += accuracy_score(labels.cpu(), predictions, normalize=False)
                        total   += images.size(0)

                        acc = round((correct / total) * 100, 4)
                        if acc > epoch_best_acc: epoch_best_acc = acc

                    # Update progress bar
                    pbar.update(1)

                # Calculate epoch accuracy
                if epoch_best_acc > self._best_acc:
                    self._best_acc =    epoch_best_acc
                    self._best_epoch =  epoch

                # Update progress bar
                pbar.set_postfix(status=colored("Complete", "green"))

                # Update model statistics
                self._accs.append(epoch_best_acc)
                self._losses.append(round(loss.item(), 4))
                self._training_report.loc[epoch-1] = [int(epoch), self._accs[-1], self._losses[-1]]

            LOGGER.info(f"Epoch {epoch:>3}/{ARGS.epochs:>3} | Accuracy: {self._accs[-1]:>8.4f} | Loss: {self._losses[-1]:>8.4f}")

    def _test_(self) -> None:
        """Conduct testing phase."""
        LOGGER.info("Commencing testing phase...")

        # Initialize progress bar
        with tqdm(total=len(self._test), desc=colored("Testing", "magenta"), leave=False, colour="cyan") as pbar:

            pbar.set_postfix(status="Testing")

            # Initialize testing metrics
            total = correct = 0

            # For image:label pairs in test data...
            for images, labels in self._test:

                # Place on GPU if available
                if torch.cuda.is_available(): images, labels = images.cuda(), labels.cuda()

                # Make predictions
                with torch.no_grad():
                    predictions = self._model(images)
                    predictions = torch.argmax(predictions, dim=1).cpu().numpy()

                    correct += accuracy_score(labels.cpu(), predictions, normalize=False)
                    total +=   images.size(0)

                # Update progress bar
                pbar.update(1)

            # Calculate accuracy
            self._test_acc = round((correct / total) * 100, 4)
            self._training_report.loc[ARGS.epochs + 1] = ['Test', 'Accuracy', self._test_acc]
            LOGGER.info(f"Test Accuracy: {self._test_acc}")

        # Update training report
        self._training_report.loc[ARGS.epochs + 2] = ['Best', self._best_acc, f'@ Epoch {self._best_epoch}']
        LOGGER.info(f"Best accuracy of {self._best_acc}% @ epoch {self._best_epoch}")