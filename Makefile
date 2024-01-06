CC = g++

all: qr_demo

qr_demo: qr_main.cpp
	$(CC) --std=c++11 qr_main.cpp  utils.cpp -o qr_demo -lpoplar -lpoputil -lpoplin -lpoprand -lboost_program_options

clean:
	rm qr_demo
