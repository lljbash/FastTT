CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
DEBUG ?= 0
DEBUG_FLAGS = -Og -DDEBUG -g
RELEASE_FLAGS = -O3 -DNDEBUG -funroll-loops -march=native -Xpreprocessor -fopenmp
OUTPUT_OPTION=-MMD -MP -o $@
LDLIBS = -lxerus -lxerus_misc
EXEC = test

SRC = $(wildcard *.cc)
OBJ = $(SRC:.cc=.o)
DEP = $(SRC:.cc=.d)

ifeq ($(DEBUG), 1)
	CXXFLAGS += $(DEBUG_FLAGS)
else
	CXXFLAGS += $(RELEASE_FLAGS)
endif

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

-include $(DEP)

clean:
	-rm -f $(OBJ) $(DEP) $(EXEC)

