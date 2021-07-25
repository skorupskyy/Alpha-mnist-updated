import mnist_loader as ml
import network1 as nw

def main():
    training_data, validation_data, test_data = ml.load_data_wrapper()
    net = nw.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


    # hidden_layers_topologies = [15, 25, [30, 10], 60, 80]
    # hidden_layers_topologies_small = ["15", "20"]
    # print(results)
    # results = []

    # for i in hidden_layers_topologies_small:
    #     net = Network([784, i, 10])
    #     res = net.SGD(training_data, 2, 10, 3.0, test_data=test_data)
    #     results.append(res)

main()