CC = gcc
CFLAGS = -Wall -Wextra -g -I src -std=c99
LDFLAGS = 
LDLIBS = 

BUILD_DIR = build
SRC_DIR = src
BUILD_MODE = debug

# Explicitly declare your main executables here
MAIN_EXES = main

# Find all source files
SRCS = $(wildcard $(SRC_DIR)/*.c)
TEST_SRCS = $(wildcard $(SRC_DIR)/*_test.c)
MAIN_SRCS = $(foreach exe,$(MAIN_EXES),$(SRC_DIR)/$(exe).c)
LIB_SRCS = $(filter-out $(TEST_SRCS) $(MAIN_SRCS), $(SRCS))

# Generate object and executable names
BUILD_TARGET_DIR = $(BUILD_DIR)/$(BUILD_MODE)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_TARGET_DIR)/%.o, $(SRCS))
LIB_OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_TARGET_DIR)/%.o, $(LIB_SRCS))
TEST_OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_TARGET_DIR)/%.o, $(TEST_SRCS))
MAIN_OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_TARGET_DIR)/%.o, $(MAIN_SRCS))
TEST_EXES = $(patsubst $(SRC_DIR)/%_test.c, $(BUILD_TARGET_DIR)/%_test, $(TEST_SRCS))
MAIN_EXE_TARGETS = $(patsubst %,$(BUILD_TARGET_DIR)/%,$(MAIN_EXES))

.PHONY: all test clean debug release build

# Default target
all: debug

# Build modes
debug: 
	@$(MAKE) BUILD_MODE=debug CFLAGS="$(CFLAGS) -DDEBUG -O0" build

release:
	@$(MAKE) BUILD_MODE=release CFLAGS="$(CFLAGS) -DNDEBUG -O2" build

# Internal build target - only build tests in debug mode
build: $(LIB_OBJS) $(MAIN_EXE_TARGETS) $(if $(filter debug,$(BUILD_MODE)),$(TEST_EXES))

# Compile source files to object files
$(BUILD_TARGET_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Link main executables
$(BUILD_TARGET_DIR)/%: $(BUILD_TARGET_DIR)/%.o $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Link test executables - corrected to include library objects
$(BUILD_TARGET_DIR)/%_test: $(BUILD_TARGET_DIR)/%_test.o $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Run tests
test: debug
	@failed=0; \
	for exe in $(TEST_EXES); do \
		echo "Running $$exe"; \
		if ./$$exe; then \
			echo "✓ $$exe passed"; \
		else \
			echo "✗ $$exe failed"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "All tests passed!"; \
	else \
		echo "$$failed test(s) failed!"; \
		exit 1; \
	fi

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Automatic dependency generation
ifeq ($(BUILD_MODE),debug)
-include $(LIB_OBJS:.o=.d) $(MAIN_OBJS:.o=.d) $(TEST_OBJS:.o=.d)
else
-include $(LIB_OBJS:.o=.d) $(MAIN_OBJS:.o=.d)
endif

# Generate dependency files
$(BUILD_TARGET_DIR)/%.d: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -MM -MT $(patsubst $(SRC_DIR)/%.c,$(BUILD_TARGET_DIR)/%.o,$<) -MT $@ $< > $@
