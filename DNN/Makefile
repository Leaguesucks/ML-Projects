CXX = g++

.SUFFIXES:

CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -Iinclude -O2 -MMD -MP

TARGET = bin/main.exe

# Discover implementations in src/ and any nested directories.
SRC = $(shell find src -type f -name '*.cpp')

OBJ = $(patsubst src/%.cpp,.out/%.o,$(SRC))
DEP = $(OBJ:.o=.d)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@

.out/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ) $(DEP)
	rm -f $(TARGET)
	rm -f .out/*.out
	rm -f .out/*.png
	rm -f training/*.png
	rm -f mnist_results/*.png

-include $(DEP)
