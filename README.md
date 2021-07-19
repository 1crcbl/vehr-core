# VEHR-CORE

A library for developing heuristics algorithms for solving vehicle routing problem (VRP) and its variants in Rust programming language.

The library has implementations of some fundamental data structures and operations that are often required in development of (meta)-heuristics algorithms for VRP. The primary aim of the library are twofold:
- A clear and easy-to-use API for tour representations and tour operations
- An efficient implementation of these operations

## Introduction
In the [first attempt](https://github.com/1crcbl/cykl-rs) in developing a solver for solving TSP and VRP, I have implemented many fundamental functionalities. Early benchmarks show a promising result on tour operations. However, such architecture seems to be quite rigid when I try to use it on other algorithm families (such as Destroy/Rebuild schemes). In these experiments, I find out that the algorithm implementation part is not as tricky and convoluted as the tour manipulation part.

Learning from this experience, I take a step back and study several algorithm families and their open-source implementations (if available). Solving VRP and its variants is very hard. Many algorithms have been proposed and tested against many problem instances. While some algorithms outperform other in many general cases, certain algorithms are designed just to solve certain problem instances.

Reading through the literature and codes, it dawns onto me that each implementation has its own tour representation and the algorithmic parts are often mixed with tour operation parts. Such approach reduces the reusability of the underlying data structure layers. Thus, if we want to develop a new kind of algorithms, we often have to start again from scratch.

This is when I start this project with the aim to abstract the underlying fundamental parts, thus removing the entanglement between algorithms and data structures.

At the moment, many core functionalities for manipulating nodes in tours and routes have been implemented and one can already develop algorithms upon this library. However, it still lacks documentations and examples. Once these issues are resolved, a Rust crate will be created. But for now, you can use the library by adding the following line to the `Cargo.toml` file:

```
[dependencies]
vehr_core = { git = "https://github.com/1crcbl/vehr-core" }
```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
