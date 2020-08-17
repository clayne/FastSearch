# FastSearch
Linear search an array of any basic type (integer or float) up to 22x faster than the typical naive loop. Requires x86 and AVX2.

Note that this also means the crossover point at which accelerated structures such as hash tables or trees become preferable to linear arrays is much, MUCH larger than usually taught or seen in the wild, especially for small elements.
