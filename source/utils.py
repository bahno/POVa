from matplotlib.pylab import plt
import numpy as np

def plotLoss(trainLoss : [], 
            valLoss : []):
    
    epochs = len(trainLoss)
    
    plt.plot(epochs, trainLoss, label='Training Loss')
    plt.plot(epochs, valLoss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, 21, 2))

    # Display the plot
    plt.legend(loc='best')
    plt.show()
