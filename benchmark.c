// benchmark.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neural_network.h"
#include "dataset.h"

int main(void) {
    // Initialisation du générateur de nombres aléatoires
    srand(time(NULL));

    // Initialisation du réseau de neurones
    NeuralNetwork net;
    int num_hidden_layers = 3;
    int hidden_sizes[3] = {30, 30, 30};
    init_network(&net, 2, num_hidden_layers, hidden_sizes, 2);

    // Génération des données d'entraînement
    int total_points;
    double norm_factor;
    TrainingPoint *training_data = (TrainingPoint*)malloc(2 * NUM_POINTS_PER_SPIRALE * sizeof(TrainingPoint));
    generate_training_data(training_data, &total_points, &norm_factor);
    printf("Facteur de normalisation: %f\n", norm_factor);

    // Définir le nombre d'epochs à exécuter pour le benchmark
    unsigned long epochs = 1000;  // par exemple 1000 epochs

    // Démarrer la mesure du temps
    clock_t start_time = clock();

    // Exécution de l'entraînement sur l'ensemble des données pour un nombre fixé d'epochs
    for (unsigned long epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < total_points; i++) {
            double inputs[2] = { training_data[i].x, training_data[i].y };
            forward_propagation(&net, inputs);
            back_propagation(&net, training_data[i].target);
        }
    }

    // Mesurer le temps écoulé
    clock_t end_time = clock();
    double total_seconds = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    double avg_time_per_epoch = total_seconds / epochs;

    printf("Benchmark: %lu epochs ont pris %f secondes (soit %f secondes par epoch en moyenne).\n", epochs, total_seconds, avg_time_per_epoch);

    // Libération des ressources
    free_network(&net);
    free(training_data);
    
    return 0;
}

