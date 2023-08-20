# Regularization 
# L2 Regularization and L1 Regularization

# https://github.com/Coding-Lane/L2-Regularization


def model(X, Y, layer_dims, learning_rate =0.3, num_iteration = 30000):

    grads = {}
    cost_list = []
    m = X.shape[1]

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, parameters)

        cost = cost_function(a3, Y)
        grads = backward_propagation(X, Y, cache)

        parameters = update_parameters(parameters, grads, learning_rate)

        if ((i%1000) == 0):
            print("cost after iteration", i, " is: ", cost)
            cost_list.append(cost)

    plt.plot(cost_list)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()

    return parameters

