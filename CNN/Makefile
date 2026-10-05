CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17
AR ?= ar
AR_PATH = $(shell command -v "$(AR)" 2>/dev/null)
CMAKE ?= cmake
BUILD_DIR ?= build
CMAKE_FLAGS ?=

# CMake owns source discovery, dependency tracking, and transitive DNN/OpenMP
# requirements. Retain the compiler and flags accepted by the old Make build.
.PHONY: all configure clean
all: configure
	+$(CMAKE) --build "$(BUILD_DIR)" --target cnn cnn_main

configure:
	$(CMAKE) -S . -B "$(BUILD_DIR)" \
		"-DCMAKE_CXX_COMPILER=$(CXX)" \
		"-DCMAKE_CXX_FLAGS=$(CPPFLAGS) $(CXXFLAGS)" \
		"-DCMAKE_AR=$(if $(AR_PATH),$(AR_PATH),$(AR))" $(CMAKE_FLAGS)

clean:
	@if [ -f "$(BUILD_DIR)/CMakeCache.txt" ]; then \
		$(CMAKE) --build "$(BUILD_DIR)" --target clean; \
	fi
