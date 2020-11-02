set -e
#gcc -g -pg -pthread -std=c99 -c -O2 main.c
#gcc -g -pg -pthread -o main main.o -lm
gcc -pthread -std=c99 -c -O2 main.c
gcc -pthread -o main main.o -lm
./main
