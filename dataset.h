#ifndef DATASET_H
#define DATASET_H

#define NUM_OUTPUTS 2
#define NUM_POINTS_PER_SPIRALE 100

// Structure d'un point d'entraînement
typedef struct {
    double x;
    double y;
    double target[NUM_OUTPUTS]; // Ex. : [1, 0] pour la spirale bleue, [0, 1] pour la spirale rouge
} TrainingPoint;

// Prototype de la fonction de génération et normalisation des données
// - data : tableau de TrainingPoint alloué par l'utilisateur
// - total_points : pointeur vers un int qui recevra le nombre total de points générés
// - norm_factor : pointeur vers un double qui recevra le facteur de normalisation utilisé
void generate_training_data(TrainingPoint *data, int *total_points, double *norm_factor);

#endif 

