//! A library for developing heuristics algorithms for solving vehicle routing problem (VRP) and its variants
//! in Rust programming language.
//! 
//! The library has implementations of some fundamental data structures and operations that are often
//! required in development of (meta)-heuristics algorithms for VRP. The primary aim of the library are twofold:
//! - A clear and easy-to-use API for tour representations and tour operations
//! - An efficient implementation of these operations
//! 
//! At a high level, the library provides a few major components:
//! - Data structures representing tours and routes for VRP and its variants
//! - APIs for performing tour and route operations 
pub mod distance;
pub mod reg;
pub mod tour;
pub mod traits;
