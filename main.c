#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include "neural_network.h"
#include "dataset.h"

#define WINDOW_WIDTH 700
#define WINDOW_HEIGHT 700

// Convertit des coordonnées normalisées [-1,1] en coordonnées écran
void world_to_screen(double x, double y, int *screen_x, int *screen_y) {
    double scale = WINDOW_WIDTH / 2.0;
    *screen_x = WINDOW_WIDTH / 2 + (int)(x * scale);
    *screen_y = WINDOW_HEIGHT / 2 - (int)(y * scale);
}

// Fonction d'affichage : affiche le fond de décision et les points d'entraînement
void render(NeuralNetwork *net, TrainingPoint *data, int total_points, SDL_Renderer *renderer) {
    int i, j;
    // Efface l'écran
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    
    // Dessine le fond de décision (grille)
    int step = 5;
    for (i = 0; i < WINDOW_WIDTH; i += step) {
        for (j = 0; j < WINDOW_HEIGHT; j += step) {
            // Conversion des coordonnées pixels en coordonnées normalisées [-1,1]
            double norm_x = ((double)i - WINDOW_WIDTH / 2.0) / (WINDOW_WIDTH / 2.0);
            double norm_y = -((double)j - WINDOW_HEIGHT / 2.0) / (WINDOW_HEIGHT / 2.0);
            double inputs[2] = { norm_x, norm_y };
            forward_propagation(net, inputs);
            // Choix de la couleur selon la sortie du réseau
            double diff = net->layers[net->num_layers - 1].neurons[0].output -
                          net->layers[net->num_layers - 1].neurons[1].output;
            if (diff > 0)
                SDL_SetRenderDrawColor(renderer, 0, 0, 150, 255); // classe 1 (bleu)
            else
                SDL_SetRenderDrawColor(renderer, 150, 0, 0, 255); // classe 2 (rouge)
            SDL_Rect rect = { i, j, step, step };
            SDL_RenderFillRect(renderer, &rect);
        }
    }
    
    // Dessine les points d'entraînement (les spirales)
    for (i = 0; i < total_points; i++) {
        int sx, sy;
        world_to_screen(data[i].x, data[i].y, &sx, &sy);
        if (data[i].target[0] > data[i].target[1])
            SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255); // spirale bleue
        else
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // spirale rouge
        SDL_Rect rect = { sx - 3, sy - 3, 6, 6 };
        SDL_RenderFillRect(renderer, &rect);
    }
    
    SDL_RenderPresent(renderer);
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    // Initialisation du réseau de neurones
    NeuralNetwork net;
    int num_hidden_layers = 3;
    int hidden_sizes[3] = {30, 30, 30};
    init_network(&net, 2, num_hidden_layers, hidden_sizes, 2);
    
    // Génération des données d'entraînement
    int total_points;
    TrainingPoint *training_data = (TrainingPoint*)malloc(2 * NUM_POINTS_PER_SPIRALE * sizeof(TrainingPoint));
    double norm_factor;
    generate_training_data(training_data, &total_points, &norm_factor);
    printf("Facteur de normalisation: %f\n", norm_factor);
    
    // Initialisation de SDL2
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Erreur SDL_Init : %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window *window = SDL_CreateWindow("Apprentissage Réseau de Neurones",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "Erreur SDL_CreateWindow : %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Erreur SDL_CreateRenderer : %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    
    int quit = 0;
    SDL_Event e;
    unsigned long epoch = 0;
    int training_done = 0;
    
    // Boucle principale SDL intégrant l'apprentissage et l'affichage en temps réel
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT)
                quit = 1;
        }
        
        if (!training_done) {
            double max_delta_epoch = 0.0;
            // Un epoch complet : on parcourt tous les exemples d'entraînement
            for (int i = 0; i < total_points; i++) {
                double inputs[2] = { training_data[i].x, training_data[i].y };
                forward_propagation(&net, inputs);
                back_propagation(&net, training_data[i].target);
                
                // Mesurer le delta dans la couche de sortie pour cet exemple
                Layer *output_layer = &net.layers[net.num_layers - 1];
                for (int j = 0; j < output_layer->num_neurons; j++) {
                    double abs_delta = fabs(output_layer->neurons[j].delta);
                    if (abs_delta > max_delta_epoch)
                        max_delta_epoch = abs_delta;
                }
            }
            epoch++;
            
            // Mise à jour du titre de la fenêtre avec le numéro d'epoch et le max delta
            char title[100];
            sprintf(title, "Epoch %lu, max delta = %f", epoch, max_delta_epoch);
            SDL_SetWindowTitle(window, title);
            
            // Critère d'arrêt : si le maximum des deltas est inférieur à 1.0, on considère l'apprentissage terminé
            if (max_delta_epoch < 1.0) {
                training_done = 1;
                SDL_SetWindowTitle(window, "Apprentissage terminé !");
            }
        }
        
        render(&net, training_data, total_points, renderer);
        SDL_Delay(30);
    }
    
    free_network(&net);
    free(training_data);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}