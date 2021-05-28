from torch.utils.data import Dataset, DataLoader
import pickle

#external libraries are only used for plotting the results
try:
    from matplotlib import pyplot as plt
    import numpy as np
    plot_results = True
except:
    plot_results = False

import dlc_practical_prologue as prologue
from models import ConvNetSep, ConvNetBase, ConvNetSepAux, ConvNetBaseAux
from utils import TrainerAux, TrainerBase, MNIST_Dataset

#define the network settings
settings = {
    'lr': 7e-3,
    'batch_size': 128,
    'n_epoch': 25,
    'aux_weight': 50,
    'channels': 256,
    'final_features': 16,
    'dropout_rate': 0.25,
    'bn': True,
    'hidden_features': 256
}

#define the experiments
experiment_1 = {
    'settings_update': {
        'lr': 7e-3,
        'dropout_rate': 0.25,
        'bn': True
    },
    'model_class': ConvNetBase,
    'trainer_class': TrainerBase,
    'save_name': 'convnetbase'
}

experiment_2 = {
    'settings_update': {
        'lr': 2e-3,
        'dropout_rate': 0.25,
        'bn': True
    },
    'model_class': ConvNetBaseAux,
    'trainer_class': TrainerAux,
    'save_name': 'convnetbaseaux'
}

experiment_3 = {
    'settings_update': {
        'lr': 7e-3,
        'dropout_rate': 0.25,
        'bn': True
    },
    'model_class': ConvNetSep,
    'trainer_class': TrainerBase,
    'save_name': 'convnetsep'
}

experiment_4 = {
    'settings_update': {
        'lr': 2e-3,
        'dropout_rate': 0.25,
        'bn': True
    },
    'model_class': ConvNetSepAux,
    'trainer_class': TrainerAux,
    'save_name': 'convnetsepaux'
}

#load the data
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
dataset = MNIST_Dataset(train_input, train_target, train_classes)
val_dataset = MNIST_Dataset(test_input, test_target, test_classes)
dataloader = DataLoader(dataset, batch_size=settings['batch_size'])
val_dataloader = DataLoader(val_dataset, batch_size=settings['batch_size'])

#define the function that runs the experiments
def run(settings_update, model_class, trainer_class, save_name):
    settings.update(settings_update)
    train_losses_list = []
    train_accs_list = []
    val_losses_list = []
    val_accs_list = []
    for i in range(15):
        model = model_class(settings)
        trainer = trainer_class(model, dataloader, val_dataloader, settings)
        train_losses, train_accs, val_losses, val_accs = trainer.train()
        train_losses_list.append(train_losses)
        train_accs_list.append(train_accs)
        val_losses_list.append(val_losses)
        val_accs_list.append(val_accs)
        with open(save_name, 'wb') as f:
            pickle.dump((train_losses_list, train_accs_list, val_losses_list, val_accs_list), f)

#run them (results are saved in pickled files)
for experiment in [experiment_1, experiment_2, experiment_3, experiment_4]:
    run(**experiment)

#load the pickled files and plot the results)
if plot_results:
    maxes = []
    for name in ['convnetbase', 'convnetbaseaux', 'convnetsep', 'convnetsepaux']:
        with open(name, 'rb') as f:
            data = pickle.load(f)
            maxes.append([max([data[3][i][j] for j in range(25)]) for i in range(15)])
        mean_train_acc = [np.mean([data[1][i][j] for i in range(15)]) for j in range(25)]
        mean_val_acc = [np.mean([data[3][i][j] for i in range(15)]) for j in range(25)]
        for acc in data[1]:
            plt.plot(acc, color='blue', alpha=0.2)
        plt.plot(mean_train_acc, color='blue', label='train acc')
        for acc in data[3]:
            plt.plot(acc, color='orange', alpha=0.2)
        plt.plot(mean_val_acc, color='orange', label='val acc')
        plt.legend()
        plt.title(f'{name} training curve')
        plt.xlabel('epoch')
        plt.ylabel('accuracy, %')
        plt.savefig(f'{name}.png', dpi=500)

    plt.boxplot(maxes, labels=['ConvNetBase', 'ConvNetBaseAux', 'ConvNetSep', 'ConvNetSepAux'])
    plt.title('Maximum validation accuracy')
    plt.ylabel('accuracy, %')
    plt.savefig('summary.png', dpi=500)
