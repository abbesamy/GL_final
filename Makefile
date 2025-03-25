CC = gcc
CFLAGS = -Wall -O2 `sdl2-config --cflags`
LDFLAGS = `sdl2-config --libs`
TARGET = nn_program

all: $(TARGET)

$(TARGET): main.o neural_network.o dataset.o
	$(CC) $(CFLAGS) -o $(TARGET) main.o neural_network.o dataset.o $(LDFLAGS) -lm

main.o: main.c neural_network.h dataset.h
	$(CC) $(CFLAGS) -c main.c

neural_network.o: neural_network.c neural_network.h
	$(CC) $(CFLAGS) -c neural_network.c

dataset.o: dataset.c dataset.h
	$(CC) $(CFLAGS) -c dataset.c

clean:
	rm -f *.o $(TARGET)

benchmark: benchmark.o neural_network.o dataset.o
	$(CC) $(CFLAGS) -o benchmark benchmark.o neural_network.o dataset.o $(LDFLAGS) -lm

benchmark.o: benchmark.c neural_network.h dataset.h
	$(CC) $(CFLAGS) -c benchmark.c
