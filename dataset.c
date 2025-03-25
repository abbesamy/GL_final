#include "dataset.h"
#include <math.h>

// Paramètres pour la génération des spirales
#define STEP_T 0.2

void generate_training_data(TrainingPoint *data, int *total_points, double *norm_factor) {
    int index = 0;
    double t;
    // Génération de la spirale bleue : cible [1, 0]
    for (int i = 0; i < NUM_POINTS_PER_SPIRALE; i++) {
        t = i * STEP_T;
        data[index].x = t * cos(t);
        data[index].y = t * sin(t);
        data[index].target[0] = 1.0;
        data[index].target[1] = 0.0;
        index++;
    }
    // Génération de la spirale rouge : cible [0, 1]
    for (int i = 0; i < NUM_POINTS_PER_SPIRALE; i++) {
        t = i * STEP_T;
        data[index].x = -t * cos(t);
        data[index].y = -t * sin(t);
        data[index].target[0] = 0.0;
        data[index].target[1] = 1.0;
        index++;
    }
    *total_points = index;
    
    // Calcul du facteur de normalisation afin que les données soient dans [-1, 1]
    double max_val = 0.0;
    for (int i = 0; i < *total_points; i++) {
        if (fabs(data[i].x) > max_val)
            max_val = fabs(data[i].x);
        if (fabs(data[i].y) > max_val)
            max_val = fabs(data[i].y);
    }
    if (max_val == 0.0)
        max_val = 1.0;
    *norm_factor = max_val;
    // Normalisation des données
    for (int i = 0; i < *total_points; i++) {
        data[i].x /= max_val;
        data[i].y /= max_val;
    }
}

