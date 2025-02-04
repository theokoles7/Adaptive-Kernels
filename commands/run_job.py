"""Execute job process."""

from json                   import dump
from logging                import Logger
from os                     import makedirs

from sklearn.metrics        import accuracy_score
from termcolor              import colored
from torch                  import argmax, no_grad, Tensor
from torch.cuda             import get_device_name, is_available
from torch.nn               import Module
from torch.nn.functional    import cross_entropy
from torch.optim            import SGD
from tqdm                   import tqdm

from datasets               import Dataset, load_dataset
from models                 import load_model
from utils                  import LOGGER

def run_job(
    dataset:            str,
    model:              str,
    kernel:             str =   None,
    size:               int =   3,
    kernel_group:       int =   13,
    location:           float = 0.0,
    scale:              float = 1.0,
    batch_size:         int =   64,
    data_path:          str =   "data",
    learning_rate:      float = 1e-1,
    save_parameters:    bool =  False,
    epochs:             int =   200,
    output_path:        str =   "output",
    **kwargs
) -> dict:
    """# Execute job process.

    ## Dataset Args:
        * dataset           (str):              Dataset on which job will execute.
        * batch_size        (int, optional):    Dataset batch size. Defaults to 64.
        * data_path         (str, optional):    Path at which dataset will be downloaded/loaded. 
                                                Defaults to "./data/".
        
    ## Model Args:
        * model             (str):              Model with which job will execute.
        * learning_rate     (float, optional):  Model's optimizer learning rate. Defaults to 0.1.
        * save_parameters   (bool, optional):   Save model parameters on completion of training. 
                                                Defaults to False.
        
    ## Kernel Args
        * kernel            (str, optional):    Kernel with which model will be loaded.
        * size              (int, optional):    Kernel size (square). Defaults to 3.
        * kernel_group      (int, optional):    Kernel configuration group. Defaults to 13.
        * location          (float, optional):  Distribution location parameter. Defaults to 0.0.
        * scale             (float, optional):  Distribution scale parameter. Defaults to 1.0.
        
    ## Job Args
        * epochs            (int, optional):    Number of epochs for which model will be trained. 
                                                Defaults to 200.
        * output_path       (str, optional):    Path at which job results will be written. Defaults 
                                                to "output".
                                                
    ## Returns:
        * dict:
            * parameters:                       Job parameters.
            * epochs:                           Individual epoch statistics (training & validation accuracy & loss).
            * best_accuracy:                    Best validation accuracy achieved during training.
            * best_epoch:                       Epoch at which highest validation accuracy was achieved.
            * test_accuracy:                    Accuracy achieved in testing phase.
            * test_loss:                        Loss incured during testing phase.
    """
    # Initialize logger
    __logger__:             Logger =    LOGGER.getChild("job-process")
    
    # Ensure output path exists
    makedirs(name = f"{output_path}/{model}/{dataset}/{kernel if kernel else 'control'}", exist_ok = True)
    __logger__.debug(f"Output path: {output_path}/{model}/{dataset}/{kernel if kernel else 'control'}")
    
    # Load dataset
    _dataset_:              Dataset =   load_dataset(**locals())
    
    # Extract data loaders
    _train_, _test_ =                   _dataset_.get_loaders()
    
    print(f"Initializing model with kernel: {locals()}")
    
    # Load model
    _model_:                Module =    load_model(
                                            channels_in =   _dataset_.channels(),
                                            channels_out =  _dataset_.classes(),
                                            dim =           _dataset_.dimension(),
                                            **locals()
                                        )
    
    # Initialize optimizer
    _optimizer_:            SGD =       SGD(
                                            params =        _model_.parameters(),
                                            lr =            learning_rate,
                                            weight_decay =  5e-4,
                                            momentum =      0.9
                                        )
    
    # Define learning rate decay interval & limit
    _lr_decay_interval_:    int =       (50 if model == "vgg" else 30)
    _lr_decay_limit_:       int =       (_lr_decay_interval_ * 3) + 1
    
    # Initialize job statistics
    _job_statistics_:       dict =      {
                                            "parameters":       locals(),
                                            "epochs":           {},
                                            "best_accuracy":    0,
                                            "best_epoch":       0,
                                            "test_accuracy":    0,
                                            "test_loss":        0
                                        }
    
    # Run model from CUDA if available
    if is_available(): _model_ = _model_.cuda()
    __logger__.debug(f"Using device: {get_device_name()}")
    
    # +============================================================================================+
    # | TRAINING                                                                                   |
    # +============================================================================================+
    
    # For each epoch prescribed
    for epoch in range(1, epochs + 1):
        
        # Set new kernels if using distribution kernel
        if kernel is not None: _model_.set_kernels(epoch = epoch, size = size)
        
        # Initialize progress bar
        with tqdm(
            total =     len(_train_) + len(_test_),
            desc =      f"Epoch {epoch}/{epochs}",
            leave =     False,
            colour =    "magenta"
        ) as training_progress:
    
            # Place model in training mode
            _model_.train()
            
            # Set progress bar status
            training_progress.set_postfix(status = colored("Training", "magenta"))
            
            # Initialize validation statistics
            total = correct = 0
            
            # For each image:label batch in train data...
            for images, labels in _train_:
                
                # Place on GPU if available
                if is_available(): images, labels = images.cuda(), labels.cuda()
                
                # Make predictions
                predictions:    Tensor =    _model_(images)
                    
                # Update correct stat
                correct +=  accuracy_score(y_true = labels.cpu(), y_pred = argmax(input = predictions, dim = 1).cpu().numpy(), normalize = False)
                
                # Update total stat
                total +=    images.size(0)
                
                # Calculate loss
                loss:           Tensor =    cross_entropy(input = predictions, target = labels)
                
                # Reset gradients
                _optimizer_.zero_grad()
                
                # Back propagation
                loss.backward()
                
                # Update weights
                _optimizer_.step()
                
                # Update progress bar
                training_progress.update(1)
                    
            # Calculate accuracy
            accuracy:   float = round((correct / total) * 100, 4)
            
            # Update job statistics for epoch
            _job_statistics_["epochs"].update({epoch: {"train_accuracy": accuracy, "train_loss": loss.item()}})

    # +============================================================================================+
    # | VALIDATION                                                                                 |
    # +============================================================================================+
            
            # Set progress bar status
            training_progress.set_postfix(status = colored("Validating", "yellow"))
            training_progress.colour = "yellow"

            # Place model in evaluation mode
            _model_.eval()
            
            # Initialize validation statistics
            total = correct = 0
            
            #For image:label batches in test data...
            for images, labels in _test_:
                
                # Place on GPU if available
                if is_available(): images, labels = images.cuda(), labels.cuda()
                
                # With no gradients calulcated...
                with no_grad():
                    
                    # Make predictions
                    predictions:    Tensor =    _model_(images)
                    
                    # Update correct stat
                    correct +=      accuracy_score(y_true = labels.cpu(), y_pred = argmax(input = predictions, dim = 1).cpu().numpy(), normalize = False)
                    
                    # Update total stat
                    total +=        images.size(0)
                
                    # Calculate loss
                    loss:           Tensor =    cross_entropy(input = predictions, target = labels)
                
                # Update progress bar
                training_progress.update(1)
                    
            # Calculate accuracy
            accuracy:   float = round((correct / total) * 100, 4)
                
        # If new accuracy record is made...
        if accuracy > _job_statistics_["best_accuracy"]:
            
            # Update record
            _job_statistics_["best_accuracy"] = accuracy
            _job_statistics_["best_epoch"] =    epoch
            
        # Update job statistics for epoch
        _job_statistics_["epochs"][epoch].update({"validation_accuracy": accuracy, "validation_loss": loss.item()})
                
        # Log epoch results
        __logger__.info(f"EPOCH {epoch}/{epochs}: {_job_statistics_['epochs'][epoch]}")
        
    # Log training record
    __logger__.info(f"TRAINING BEST ACCURACY: {_job_statistics_['best_accuracy']} @ EPOCH {_job_statistics_['best_epoch']}")
    
    # +============================================================================================+
    # | TESTING                                                                                    |
    # +============================================================================================+
        
    # Initialize progress bar
    with tqdm(
        total =     len(_test_),
        desc =      f"Testing",
        leave =     False,
        colour =    "cyan"
    ) as testing_progress:
        
        # Set progress bar status
        training_progress.set_postfix(status = colored("Testing", "green"))
            
        # Initialize validation statistics
        total = correct = 0
        
        #For image:label batches in test data...
        for images, labels in _test_:
            
            # Place on GPU if available
            if is_available(): images, labels = images.cuda(), labels.cuda()
            
            # With no gradients calulcated...
            with no_grad():
                
                # Make predictions
                predictions:    Tensor =    _model_(images)
                
                # Calculate loss
                loss:           Tensor =    cross_entropy(input = predictions, target = labels)
                
                # Update correct stat
                correct +=  accuracy_score(y_true = labels.cpu(), y_pred = argmax(input = predictions, dim = 1).cpu().numpy(), normalize = False)
                
                # Update total stat
                total +=    images.size(0)
            
            # Update progress bar
            testing_progress.update(1)
            
        # Close progress bar, in case there were not an even number of batches
        testing_progress.close()
                
        # Calculate accuracy
        accuracy:   float = round((correct / total) * 100, 4)
        
        # Update job statistics
        _job_statistics_.update({
            "test_accuracy":    accuracy,
            "test_loss":        loss.item()
        })
        
    # Log testing statistics
    __logger__.info(f"TEST ACCURACY: {_job_statistics_['test_accuracy']}, TEST LOSS: {_job_statistics_['test_loss']}")
    
    # Save job statistics to file
    with open(
        file =      f"{output_path}/{model}/{dataset}/{kernel if kernel is not None else 'control'}.json", 
        mode =      "w", 
        encoding =  "utf-8"
    ) as file_out: dump(obj = _job_statistics_, fp = file_out, indent = 2, default = str)
        
    
    # Return job statistics
    return _job_statistics_