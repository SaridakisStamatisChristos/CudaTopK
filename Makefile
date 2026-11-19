CXX ?= c++
CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic
BUILD ?= build

CPPFLAGS += -Iinclude
HOST_FLAGS = $(CXXFLAGS) -std=c++17

.PHONY: all test clean
all: $(BUILD)/libcudatopk.a
test: $(BUILD)/test_cpu
	$(BUILD)/test_cpu

$(BUILD):
	mkdir -p $@

$(BUILD)/cpu.o: src/cpu.cpp include/cudatopk/topk.hpp src/key.hpp | $(BUILD)
	$(CXX) $(CPPFLAGS) $(HOST_FLAGS) -c $< -o $@

$(BUILD)/libcudatopk.a: $(BUILD)/cpu.o
	$(AR) rcs $@ $^

$(BUILD)/test_cpu: tests/test_cpu.cpp tests/cases.hpp $(BUILD)/libcudatopk.a
	$(CXX) $(CPPFLAGS) $(HOST_FLAGS) $< $(BUILD)/libcudatopk.a $(LDFLAGS) -o $@

clean:
	rm -rf -- "$(BUILD)"
