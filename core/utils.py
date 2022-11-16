import itertools
import matplotlib.pyplot as plt
import numpy
import torch 
import torchvision


def np_to_tb(array):
    # if 2D :
    if array.ndim == 2:
        # HW => CHW
        array = numpy.expand_dims(array,axis=0)
        # CHW => NCHW
        array = numpy.expand_dims(array,axis=0)
    elif array.ndim == 3:
        # HWC => CHW
        array = array.transpose(2, 0, 1)
        # CHW => NCHW
        array = numpy.expand_dims(array,axis=0)
    
    array = torch.from_numpy(array)
    array = torchvision.utils.make_grid(array, normalize=True, scale_each=True)
    return array


def plot_confusion_matrix(conf_matrix, target_names, title='Confusion Matrix', cmap=None, normalize=True, save_name=None):
    accuracy = numpy.trace(conf_matrix) / float(numpy.sum(conf_matrix))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('coolwarm')

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = numpy.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=25)
        plt.yticks(tick_marks, target_names)

    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, numpy.newaxis]

    thresh = conf_matrix.max() / 1.5 if normalize else conf_matrix.max() / 2
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(conf_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(conf_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if    save_name is None: plt.show()
    else: plt.savefig(save_name)
    plt.close()


def segmentation(horizon, ground_truth, prediction):
    pass


if __name__ == '__main__':
    conf_matrix_c1 = numpy.array([
        [2.7846863e+07, 6.1257200e+05, 6.2382000e+04, 7.4000000e+01, 2.8100000e+02, 0.0000000e+00],
	    [1.0599040e+06, 1.1063671e+07, 1.0536910e+06, 6.8860000e+03, 6.8800000e+02, 0.0000000e+00],
	    [6.1160800e+05, 1.2688900e+06, 6.5284966e+07, 1.7635900e+05, 1.5981000e+04, 9.9200000e+03],
	    [1.1588000e+04, 1.8803100e+05, 1.2040920e+06, 5.6500510e+06, 9.1276800e+05, 6.9142000e+04],
        [2.0360000e+03, 1.6314000e+04, 4.1993300e+06, 8.6998800e+05, 7.4116560e+06, 1.9952000e+05],
	    [6.9100000e+02, 1.1720000e+03, 1.0875970e+06, 1.4285200e+05, 7.4797100e+05, 1.0144650e+06]
    ])
    conf_matrix_c3 = numpy.array([
        [2.8076048e+07, 3.7117000e+05, 7.2605000e+04, 2.3470000e+03, 2.0000000e+00, 0.0000000e+00],
        [1.5479990e+06, 1.0821899e+07, 7.9875600e+05, 1.5955000e+04, 2.2900000e+02, 2.0000000e+00],
        [1.4460410e+06, 1.9744050e+06, 6.3646389e+07, 2.4803700e+05, 3.3038000e+04, 1.9814000e+04],
        [2.9495000e+04, 2.0476500e+05, 9.4199200e+05, 5.4294610e+06, 1.1717360e+06, 2.5822300e+05],
        [2.0084000e+04, 9.5033000e+04, 4.4762020e+06, 1.1161860e+06, 6.6043820e+06, 3.8695700e+05],
        [2.0500000e+03, 2.3050000e+03, 8.6699800e+05, 1.4227700e+05, 4.3030900e+05, 1.5508090e+06]
    ])

    target_names = ['Upper NS', 'Middle NS', 'Lower NS', 'Rijnland + Chalk', 'Scruff', 'Zechstein']
    plot_confusion_matrix(conf_matrix_c1, target_names, save_name='runs/conf_matrix_c1.pdf')
    plot_confusion_matrix(conf_matrix_c3, target_names, save_name='runs/conf_matrix_c3.pdf')
    