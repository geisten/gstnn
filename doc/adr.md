# The gstnn architecture

The architecture is designed to run with the least possible number of external libraries and modules. It is intended to
be as minimalistic as possible to keep the complexity low and thus the clarity high. It therefore also uses the POSIX
Unix principles to exchange data.

                                      ┌──────┐
                                      │ Kern │
                                      └──────┘
                                          ▲
                                          │
                   ┌─────────┐       ┌────┴──────┐
                   │  gstnn  ├──────►│ config.h  │
                   └─────────┘       └───────────┘

The gstnn program architecture is build on three main components: _kern_, _gstnn_ and _config.h_ The _kern_ contains the
functionalities to compute and train Deep Learning algorithms. The _gstnn_ module contains the user interface to run the
created model (controller). In _config.h_ the models can be created and configured, which are then executed within the
module _gstnn.c_.

## Architectural Decision Records

A List of accepted architectural decision records. The decisions are absolute. This means that once a decision is
changed, the original architecture must first be changed until the new architecture decision takes effect. The
transition period must be clearly specified, and the existing architecture must be promptly refactored.

### API Design: Always Pass struct by value as function parameter- 07.06.2020

The API places particular emphasis on clear semantics to make the code readable and maintainable:
Passing by value makes it clear that the value ist constant and only the copy is changed.

#### Consequences

On a typical PC, performance should not be an issue even for fairly large structures. Other criteria are more important,
especially semantics: Do you indeed want to work on a copy? Or on the same object, e.g. when manipulating linked lists?
The guideline should be to express the desired semantics with the most appropriate language construct in order to make
the code readable and maintainable.

### API Design: Never point to another object within a structure - 07.06.2020

A structure should never point to another structure or other data (except arrays). References to other objects are made
using a unique ID (identifier).

#### Consequences

It may be more complex to determine a referenced object by its id. However, it minimizes side effects.

