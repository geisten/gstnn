Style

Note that the following are guidelines and the most important aspect of style is consistency. Strive to keep your style
consistent with the project on which you are working. It is up to the project maintainer to take some liberty in the
style guidelines. Recommended Reading

The following contain good information, some of which is repeated below, some of which is contradicted below.

    https://man.openbsd.org/style
    http://doc.cat-v.org/bell_labs/pikestyle
    https://www.kernel.org/doc/Documentation/process/coding-style.rst

File Layout

    Comment with LICENSE and possibly short explanation of file/tool.
    Headers
    Macros
    Types
    Function definitions grouped/ordered in logical manner.
    main

C Features

    Use C99 without extensions (ISO/IEC 9899:1999).
    Use POSIX.1-2008:
        When using gcc define _POSIX_C_SOURCE 200809L.
        Alternatively define _XOPEN_SOURCE 700.
    Variadic macros are acceptable, but remember:
        __VA_ARGS__ not a named parameter.
        Arg list cannot be empty.

Blocks

    All variable declarations at top of block.
    { on same line preceded by single space.
    } on own line unless continuing statement (if else, do while, ...).

Use block for single statement if inner statement needs a block.

```c
for (;;) {
	if (foo) {
		bar;
		baz;
	}
}
```

Use block if another branch of the same statement needs a block:

```c
if (foo) {
	bar;
} else {
	baz;
	qux;
}
```

Leading Whitespace

Use tabs for indentation and spaces for alignment. This ensures everything will line up independent of tab size. This
means:

    No tabs except beginning of line.
    Use spaces - not tabs - for multiline macros as the indentation level is 0, where the #define began.

Functions
    
    Functions not used outside translation unit should be declared and defined static.

Example:

```c
static void usage(void) {
	eprintf("usage: %s [file ...]\n", argv0);
}
```

Variables

    Global variables not used outside translation unit should be declared static.
    In declaration of pointers the * is adjacent to variable name, not type.

Keywords

    Use a space after if, for, while, switch (they are not function calls).
    Do not use a space after the opening ( and before the closing ).
    Use () with sizeof.
    Do not use a space with sizeof().

Switch

    Do not indent cases another level.
    Comment cases that FALLTHROUGH.

Example:

```c
switch (value) {
case 0: /* FALLTHROUGH */
case 1:
case 2:
	break;
default:
	break;
}
```

Headers

    Place system/libc headers first in alphabetical order.
        If headers must be included in a specific order add a comment to explain.
    Place local headers after an empty line.
    When writing and using local headers.
        Try to avoid cyclic header inclusion dependencies.
        Instead ensure they are included where and when they are needed.
        Read https://talks.golang.org/2012/splash.article#TOC_5.
        Read http://plan9.bell-labs.com/sys/doc/comp.html

User Defined Types

    Do not use type_t naming (it is reserved for POSIX and less readable).

Line Length

    Keep lines to reasonable length (max 80 characters).

Handling Errors

    When functions return -1 for error test against 0 not -1:

```c
if (func() < 0)
	hcf();
```

    Use goto to unwind and cleanup when necessary instead of multiple nested levels.
    return or exit early on failures instead of multiple nested levels.
    Unreachable code should have a NOTREACHED comment.
    Think long and hard on whether or not you should cleanup on fatal errors. For simple "one-shot" programs (not daemons) it can be OK to not free memory. It is advised to cleanup temporary files however.

Enums and #define

Use enums for values that are grouped semantically and #define otherwise:

```c
#define MAXSZ  4096
#define MAGIC1 0xdeadbeef

enum {
	DIRECTION_X,
	DIRECTION_Y,
	DIRECTION_Z
};
```

