#include "neural_network.h"
#include <stdlib.h>
#include <stdio.h>

// Retourne un poids aléatoire entre -1 et 1
static double random_weight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// Fonction d'activation tanh
double activation(double x) {
    return tanh(x);
}

// Dérivée de tanh : 1 - tanh(x)²
double activation_derivative(double output) {
    return 1.0 - output * output;
}

// Création d'un neurone sans biais
Neuron create_neuron(int num_inputs) {
    Neuron n;
    n.num_inputs = num_inputs;
    n.weights = (double*)malloc(num_inputs * sizeof(double));
    for (int i = 0; i < num_inputs; i++) {
        n.weights[i] = random_weight();
    }
    n.output = 0.0;
    n.delta = 0.0;
    return n;
}

// Création d'une couche contenant "num_neurons" neurones, chacun ayant "num_inputs" entrées
Layer create_layer(int num_neurons, int num_inputs) {
    Layer layer;
    layer.num_neurons = num_neurons;
    layer.neurons = (Neuron*)malloc(num_neurons * sizeof(Neuron));
    for (int i = 0; i < num_neurons; i++) {
        layer.neurons[i] = create_neuron(num_inputs);
    }
    return layer;
}

// Initialisation du réseau
// num_inputs : nombre de neurones de la couche d'entrée
// num_hidden_layers : nombre de couches cachées
// hidden_sizes[] : tableau des tailles des couches cachées
// num_outputs : nombre de neurones de la couche de sortie
void init_network(NeuralNetwork *net, int num_inputs, int num_hidden_layers, int hidden_sizes[], int num_outputs) {
    int total_layers = 1 + num_hidden_layers + 1;
    net->num_layers = total_layers;
    net->layers = (Layer*)malloc(total_layers * sizeof(Layer));

    // Couche d'entrée (aucun poids, on stocke simplement les entrées)
    net->layers[0].num_neurons = num_inputs;
    net->layers[0].neurons = (Neuron*)malloc(num_inputs * sizeof(Neuron));
    for (int i = 0; i < num_inputs; i++) {
        net->layers[0].neurons[i].num_inputs = 0;
        net->layers[0].neurons[i].weights = NULL;
        net->layers[0].neurons[i].output = 0.0;
        net->layers[0].neurons[i].delta = 0.0;
    }

    // Première couche cachée
    net->layers[1] = create_layer(hidden_sizes[0], num_inputs);

    // Autres couches cachées
    for (int i = 2; i <= num_hidden_layers; i++) {
        net->layers[i] = create_layer(hidden_sizes[i - 1], hidden_sizes[i - 2]);
    }

    // Couche de sortie : nombre d'entrées = taille de la dernière couche cachée (ou num_inputs si pas de couche cachée)
    int index_output = total_layers - 1;
    int last_hidden_size = (num_hidden_layers > 0) ? hidden_sizes[num_hidden_layers - 1] : num_inputs;
    net->layers[index_output] = create_layer(num_outputs, last_hidden_size);
}

// Propagation avant : calcule la sortie du réseau pour un ensemble d'entrées
void forward_propagation(NeuralNetwork *net, double inputs[]) {
    // La couche d'entrée reçoit directement les valeurs d'entrée
    for (int i = 0; i < net->layers[0].num_neurons; i++) {
        net->layers[0].neurons[i].output = inputs[i];
    }
    // Propagation à travers les couches suivantes
    for (int l = 1; l < net->num_layers; l++) {
        Layer *prev_layer = &net->layers[l - 1];
        Layer *current_layer = &net->layers[l];
        for (int j = 0; j < current_layer->num_neurons; j++) {
            Neuron *n = &current_layer->neurons[j];
            double sum = 0.0;
            for (int k = 0; k < n->num_inputs; k++) {
                sum += prev_layer->neurons[k].output * n->weights[k];
            }
            n->output = activation(sum);
        }
    }
}

// Rétropropagation : ajuste les poids en fonction de l'erreur
void back_propagation(NeuralNetwork *net, double target[]) {
    int l, i, j;
    // Calcul du delta pour la couche de sortie
    Layer *output_layer = &net->layers[net->num_layers - 1];
    for (i = 0; i < output_layer->num_neurons; i++) {
        Neuron *n = &output_layer->neurons[i];
        double error = target[i] - n->output;
        n->delta = error * activation_derivative(n->output);
    }
    // Propagation de l'erreur vers les couches cachées
    for (l = net->num_layers - 2; l >= 1; l--) {
        Layer *current_layer = &net->layers[l];
        Layer *next_layer = &net->layers[l + 1];
        for (i = 0; i < current_layer->num_neurons; i++) {
            double error = 0.0;
            for (j = 0; j < next_layer->num_neurons; j++) {
                error += next_layer->neurons[j].weights[i] * next_layer->neurons[j].delta;
            }
            current_layer->neurons[i].delta = error * activation_derivative(current_layer->neurons[i].output);
        }
    }
    // Mise à jour des poids (pour toutes les couches sauf la couche d'entrée)
    for (l = 1; l < net->num_layers; l++) {
        Layer *prev_layer = &net->layers[l - 1];
        Layer *current_layer = &net->layers[l];
        for (i = 0; i < current_layer->num_neurons; i++) {
            Neuron *n = &current_layer->neurons[i];
            for (j = 0; j < n->num_inputs; j++) {
                n->weights[j] += LEARNING_RATE * n->delta * prev_layer->neurons[j].output;
            }
        }
    }
}

// Libération de la mémoire allouée au réseau
void free_network(NeuralNetwork *net) {
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        if (l > 0) {
            for (int i = 0; i < layer->num_neurons; i++) {
                free(layer->neurons[i].weights);
            }
        }
        free(layer->neurons);
    }
    free(net->layers);
}