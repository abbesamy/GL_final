#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.01

// Structure d'un neurone (sans biais)
typedef struct {
    int num_inputs;      // nombre d'entrées
    double *weights;     // poids associés aux entrées
    double output;       // sortie du neurone
    double delta;        // delta utilisé pour la rétropropagation
} Neuron;

// Structure d'une couche de neurones
typedef struct {
    int num_neurons;
    Neuron *neurons;
} Layer;

// Structure d'un réseau de neurones
typedef struct {
    int num_layers;      // nombre total de couches (entrée + cachées + sortie)
    Layer *layers;       // tableau dynamique de couches
} NeuralNetwork;

// Prototypes des fonctions du réseau
Neuron create_neuron(int num_inputs);
Layer create_layer(int num_neurons, int num_inputs);
void init_network(NeuralNetwork *net, int num_inputs, int num_hidden_layers, int hidden_sizes[], int num_outputs);
void forward_propagation(NeuralNetwork *net, double inputs[]);
void back_propagation(NeuralNetwork *net, double target[]);
void free_network(NeuralNetwork *net);

// Fonctions d'activation et leur dérivée
double activation(double x);
double activation_derivative(double output);

#endif // NEURAL_NETWORK_H