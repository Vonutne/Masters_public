CXX = nvcc -O3 -arch=sm_80
# --extra-device-vectorization 
#-arch=sm_60

SRC = main.cu
HELPERS = kernels.cu.h
EXECUTABLE =conv

default: compile run

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE) 2048 32 32
	./$(EXECUTABLE) 4096 64 64
	./$(EXECUTABLE) 1500 22 22
	./$(EXECUTABLE) 2048 1 32
	./$(EXECUTABLE) 4096 64 64
	./$(EXECUTABLE) 4096 32 32
	./$(EXECUTABLE) 4096 8 8
	./$(EXECUTABLE) 1028 32 32
	./$(EXECUTABLE) 2052 64 64	
#	./$(EXECUTABLE) 4096 1 1

clean:
	rm -f $(EXECUTABLE)

