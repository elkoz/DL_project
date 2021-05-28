import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

class MNIST_Dataset(Dataset):
    #subclass of pytorch Dataset that handles our data
    def __init__(self, input_data, target, classes):
        super(MNIST_Dataset, self).__init__()
        self.X = input_data
        self.y = target
        self.classes = classes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {'input': self.X[index], 'target': self.y[index], 'classes': self.classes[index]}


class Trainer():
    # parent class for trainers
    def __init__(self, model, dataloader, val_dataloader, settings):
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.lr = settings['lr']
        self.n_epoch = settings['n_epoch']

    def loss_function(self, batch, total, correct):
        pass

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(self.n_epoch):
            self.model.train()
            epoch_loss = 0
            aux_epoch_loss = 0
            total = 0
            correct = 0

            for batch in self.dataloader:
                loss, total, correct, aux_loss = self.loss_function(batch, total, correct)
                epoch_loss += loss.item()
                aux_epoch_loss += aux_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = 100 * correct / total
            epoch_loss = epoch_loss / len(self.dataloader)
            aux_epoch_loss = aux_epoch_loss / len(self.dataloader)
            val_acc, val_loss, val_aux = self.evaluate()

            train_losses.append(epoch_loss)
            train_accs.append(acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(
                f'[epoch {epoch}]: accuracy {acc:.2f}, loss {epoch_loss:.2f}, val accuracy {val_acc:.2f}, val loss {val_loss:.2f}')
            if aux_epoch_loss != 0:
                print(f'                 aux loss {aux_epoch_loss:.2f}, val aux loss {val_aux:.2f}')

        return train_losses, train_accs, val_losses, val_accs

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        aux_epoch_loss = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                loss, total, correct, aux_loss = self.loss_function(batch, total, correct)
                epoch_loss += loss.item()
                aux_epoch_loss += aux_loss.item()

        acc = 100 * correct / total
        epoch_loss = epoch_loss / len(self.val_dataloader)
        aux_epoch_loss = aux_epoch_loss / len(self.val_dataloader)

        return acc, epoch_loss, aux_epoch_loss


class TrainerBase(Trainer):
    #subclass that handles models without auxillary loss
    def __init__(self, model, dataloader, val_dataloader, settings):
        super(TrainerBase, self).__init__(model, dataloader, val_dataloader, settings)
        self.bce = nn.BCELoss(reduction='mean')

    def loss_function(self, batch, total, correct):
        batch_input = batch['input'].float()
        batch_target = batch['target'].unsqueeze(1).float()

        predicted = self.model(batch_input)

        loss = self.bce(predicted, batch_target)

        predicted = torch.round(predicted)
        correct += torch.sum(batch_target == predicted)
        total += len(batch_target)

        return loss, total, correct, torch.tensor(0)


class TrainerAux(Trainer):
    # subclass that handles models with auxillary loss
    def __init__(self, model, dataloader, val_dataloader, settings):
        super(TrainerAux, self).__init__(model, dataloader, val_dataloader, settings)
        self.bce = nn.BCELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.a = settings['aux_weight']

    def loss_function(self, batch, total, correct):
        batch_input = batch['input'].float()
        batch_target = batch['target'].unsqueeze(1).float()
        batch_classes = batch['classes'].flatten()

        predicted, classes = self.model(batch_input)

        loss = self.bce(predicted, batch_target)

        predicted = torch.round(predicted)
        correct += torch.sum(batch_target == predicted)
        total += len(batch_target)

        aux_loss = self.a * self.ce(classes, batch_classes)
        loss += aux_loss

        return loss, total, correct, aux_loss
