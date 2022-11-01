#=====================================================================
# Build, test and install the geisten neural network program
#
# Typing 'make' or 'make help' will print the available make commands
#
#======================================================================

PROJECT_NAME = geisten
PREFIX ?= /usr/local

MKDIR_P ?= mkdir -p
RM ?= rm

#--------------------------- DON'T change this part ----------------------------

src = $(wildcard verify/*.c) $(wildcard test/*.c)
obj = $(src:.c=.o)
dep = $(obj:.o=.d)


# CFLAGS ?= -I. -march=native -mtune=native -MP -Wall -Wextra -mavx -Wstrict-overflow -ffast-math -fsanitize=address -O3 -MMD
CFLAGS ?= -I. -march=native -mtune=native -MP -Wall -Wextra -mavx -Wstrict-overflow -ffast-math -fsanitize=address -O -MMD -g2

LDFLAGS ?= -ffast-math -lm -fsanitize=address -mavx -fopenmp -lpthread

options:
	@echo $(PROJECT_NAME) build options:
	@echo "CFLAGS   = ${CFLAGS}"
	@echo "LDFLAGS  = ${LDFLAGS}"
	@echo "CC       = ${CC}"

debug: CFLAGS+= -O0 -g3 -gdwarf -DDEBUG
debug: $(PROJECT_NAME)

all: options $(PROJECT_NAME)  ## build all binaries and libraries

%.o : %.c
	$(CC) $(CFLAGS) -c $< $(LIB_PATH) $(LIBS) -o $@ $(LDFLAGS)

# build the objewct file
$(PROJECT_NAME): $(obj)
	$(CC) -o $@ $^ $(LDFLAGS)

lib%.a: %.o stats.o  ## link geisten blas library files into a static library
	ar rcs $@ $+
	ranlib $@

# build the unit tests
test/%: test/%.o
	$(CC) -o $@ $< $(LDFLAGS)
	$@ ||  (echo "Test $^ failed" && exit 1)

test: test/test_geisten_bit ## run all test programs
	@echo "Success, all tests of project '$(PROJECT_NAME)' passed."

# build the integration tests
verify/%: verify/%.o
	$(CC) -o $@ $< $(LDFLAGS)
	#$@ -t data/mnist_train_targets.i8 data/mnist_train_images.i8  ||  (echo "Test $^ failed" && exit 1)

verify_mnist_i8: verify/mnist_i8

.PHONY: clean
# clean the build
clean:  ## cleanup - remove the target build files
	rm -f $(obj) $(dep) $(PROJECT_NAME) test/test_kern test/*.o test/*.d verify/mnist_i8

config_%: ## copy a config file to config.h
	cp $@.h config.h


.PHONY: install
install: $(PROJECT_NAME)  ## install the target build to the target directory ('$(DESTDIR)$(PREFIX)/bin')
	install $< $(DESTDIR)$(PREFIX)/bin/$(PROJECT_NAME)

.PHONY: uninstall
uninstall: ## remove the build from the target directory ('$(DESTDIR)$(PREFIX)/bin')
	rm -f $(DESTDIR)$(PREFIX)/bin/$(PROJECT_NAME)

# ----------------- TOOLS ------------------------------------------

# c header docu
doc/%.md: %.h
	$(MKDIR_P) $(dir $@)
	cat $< | awk '/\/\*\*/ {blk=1}; {if(blk) print $0}; /\*\// {blk=0}' | sed 's/..[*/ ]\?//' > $@

docs: doc/kern.md ## build the documentation of the header files in markdown format

help: ## print this help information. Type 'make all' to build the project
	@awk -F ':|##' '/^[^\t].+?:.*?##/ {\
		printf "\033[36m%-30s\033[0m %s\n", $$1, $$NF \
		}' $(MAKEFILE_LIST)


# just type 'make print-VARIABLE to get the value of the variable >>VARIABLE<<
print-%  : ; @echo $* = $($*) ## get the value of a makefile variable '%' (type make print-VARIABLE to get value of VARIABLE)

.DEFAULT_GOAL=all

-include $(dep)
